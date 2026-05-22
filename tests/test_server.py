import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts import server

MAX_TASK_WAIT_ITERATIONS = 40
TASK_WAIT_SECONDS = 0.05


class ServerTests(unittest.TestCase):
    def setUp(self):
        server.task_manager = server.TaskManager()
        server.reset_engine_cache()
        self.client = TestClient(server.app)

    def wait_for_task(self, task_id: str) -> dict:
        for _ in range(MAX_TASK_WAIT_ITERATIONS):
            response = self.client.get(f"/v1/tasks/{task_id}")
            payload = response.json()
            if payload["status"] != "running":
                return payload
            time.sleep(TASK_WAIT_SECONDS)
        self.fail("Task did not finish in time.")

    def test_index_page_is_served(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Article Writer Control Center", response.text)
        self.assertIn("Run ingest", response.text)

    def test_models_endpoint_lists_names(self):
        fake_settings = SimpleNamespace(
            create_ollama_client=lambda timeout=None: SimpleNamespace(
                list=lambda: {"models": [{"name": "alpha"}, {"model": "beta"}]}
            )
        )
        with patch.object(server.RuntimeSettings, "from_env", return_value=fake_settings):
            response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["alpha", "beta"]})

    def test_settings_defaults_endpoint_returns_expected_payload(self):
        fake_settings = SimpleNamespace(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="articles",
            ollama_base_url="http://ollama",
            ollama_embed_model="embed-model",
            ollama_chat_model="chat-model",
            grobid_base_url="http://grobid",
            chunk_sizes=(2048, 768, 256),
            vector_top_k=24,
            bm25_top_k=12,
            fused_top_k=8,
        )
        with patch.object(server.RuntimeSettings, "from_env", return_value=fake_settings):
            response = self.client.get("/v1/settings/defaults")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
                "collection_name": "articles",
                "ollama_base_url": "http://ollama",
                "ollama_embed_model": "embed-model",
                "ollama_chat_model": "chat-model",
                "grobid_base_url": "http://grobid",
                "chunk_sizes": "2048,768,256",
                "vector_top_k": 24,
                "bm25_top_k": 12,
                "fused_top_k": 8,
            },
        )

    def test_ingest_task_captures_logs_and_result(self):
        def fake_run(force, settings_overrides=None):
            print(f"force={force}")
            print(f"settings={settings_overrides['OLLAMA_CHAT_MODEL']}")
            return (2, 1)

        with patch.object(server, "_run_ingest", side_effect=fake_run):
            response = self.client.post(
                "/v1/tasks/ingest",
                json={
                    "force": True,
                    "settings_overrides": {"OLLAMA_CHAT_MODEL": "qwen-test"},
                },
            )
            task = self.wait_for_task(response.json()["id"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], [2, 1])
        self.assertIn("force=True", task["log_text"])
        self.assertIn("qwen-test", task["log_text"])

    def test_concurrent_tasks_run_one_at_a_time(self):
        first_ready = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_task():
            print("first-start")
            first_ready.set()
            if not first_release.wait(timeout=2):
                raise RuntimeError("First task was not released in time.")
            print("first-end")

        def second_task():
            second_started.set()
            print("second-only")

        first = server.task_manager.create_task("first", {}, first_task)
        second = server.task_manager.create_task("second", {}, second_task)

        self.assertTrue(first_ready.wait(timeout=2))
        time.sleep(TASK_WAIT_SECONDS)
        self.assertFalse(second_started.is_set())
        first_release.set()

        first_payload = self.wait_for_task(first.id)
        second_payload = self.wait_for_task(second.id)

        self.assertEqual(first_payload["status"], "completed")
        self.assertEqual(second_payload["status"], "completed")
        self.assertIn("first-start", first_payload["log_text"])
        self.assertIn("first-end", first_payload["log_text"])
        self.assertNotIn("second-only", first_payload["log_text"])
        self.assertIn("second-only", second_payload["log_text"])
        self.assertNotIn("first-end", second_payload["log_text"])

    def test_query_endpoint_returns_text_and_citations(self):
        class FakeResponse:
            source_nodes = [
                SimpleNamespace(
                    score=0.9,
                    node=SimpleNamespace(
                        metadata={
                            "title": "Paper",
                            "author": "Author",
                            "year": "2024",
                            "page": 2,
                            "paragraph": 4,
                        }
                    ),
                )
            ]

            def __str__(self):
                return "Grounded answer"

        fake_response = FakeResponse()
        fake_engine = SimpleNamespace(query=lambda prompt: fake_response)

        with patch.object(server, "get_engine", return_value=fake_engine):
            response = self.client.post(
                "/v1/research/query",
                json={"prompt": "What changed?", "settings_overrides": {"OLLAMA_CHAT_MODEL": "x"}},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "Grounded answer")
        self.assertEqual(payload["citations"][0]["title"], "Paper")

    def test_chat_endpoint_assembles_messages_and_parses_response(self):
        chat_calls = {}

        def fake_chat(*, model, messages):
            chat_calls["model"] = model
            chat_calls["messages"] = messages
            return {"message": {"content": "Assistant reply"}}

        def fake_from_env(overrides=None):
            chat_calls["overrides"] = overrides
            def fake_create_ollama_client(timeout=None):
                chat_calls["timeout"] = timeout
                return SimpleNamespace(chat=fake_chat)

            return SimpleNamespace(
                ollama_chat_model=overrides["OLLAMA_CHAT_MODEL"],
                create_ollama_client=fake_create_ollama_client,
            )

        with patch.object(server.RuntimeSettings, "from_env", side_effect=fake_from_env):
            response = self.client.post(
                "/v1/chat",
                json={
                    "prompt": "Summarize the findings",
                    "messages": [{"role": "system", "content": "Be concise."}],
                    "settings_overrides": {
                        "OLLAMA_CHAT_MODEL": "qwen-chat",
                        "BLANK": "   ",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(chat_calls["overrides"], {"OLLAMA_CHAT_MODEL": "qwen-chat"})
        self.assertEqual(chat_calls["timeout"], server.OLLAMA_CHAT_TIMEOUT_SECONDS)
        self.assertEqual(chat_calls["model"], "qwen-chat")
        self.assertEqual(
            chat_calls["messages"],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Summarize the findings"},
            ],
        )
        self.assertEqual(response.json()["message"], "Assistant reply")
        self.assertEqual(response.json()["messages"], chat_calls["messages"])


if __name__ == "__main__":
    unittest.main()

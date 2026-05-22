import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        server.task_manager = server.TaskManager()
        server.reset_engine_cache()
        self.client = TestClient(server.app)

    def wait_for_task(self, task_id: str) -> dict:
        for _ in range(40):
            response = self.client.get(f"/v1/tasks/{task_id}")
            payload = response.json()
            if payload["status"] != "running":
                return payload
            time.sleep(0.05)
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

        self.assertEqual(response.status_code, 200)
        task = self.wait_for_task(response.json()["id"])
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], [2, 1])
        self.assertIn("force=True", task["log_text"])
        self.assertIn("qwen-test", task["log_text"])

    def test_query_endpoint_returns_text_and_citations(self):
        fake_response = SimpleNamespace(
            __str__=lambda self: "Grounded answer",
            source_nodes=[
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
            ],
        )
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


if __name__ == "__main__":
    unittest.main()

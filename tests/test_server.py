import threading
import time
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts import build_index
from scripts import ingest
from scripts import server
from scripts import synthesis

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
        self.assertIn("Beginner model guide", response.text)
        self.assertIn("Embedding model", response.text)

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

    def test_pdf_listing_endpoint_returns_sorted_pdf_names(self):
        with TemporaryDirectory() as temporary_dir:
            pdf_dir = Path(temporary_dir) / "pdfs"
            processed_dir = Path(temporary_dir) / "processed"
            pdf_dir.mkdir()
            (pdf_dir / "zeta.pdf").write_text("", encoding="utf-8")
            (pdf_dir / "alpha.pdf").write_text("", encoding="utf-8")
            (pdf_dir / "notes.txt").write_text("", encoding="utf-8")
            fake_settings = SimpleNamespace(
                pdf_dir=pdf_dir,
                ensure_runtime_dirs=lambda: processed_dir.mkdir(exist_ok=True),
            )

            with patch.object(server.RuntimeSettings, "from_env", return_value=fake_settings):
                response = self.client.get("/v1/corpus/pdfs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"files": ["alpha.pdf", "zeta.pdf"]})

    def test_ingest_task_captures_logs_and_result(self):
        run_calls = {}

        def fake_run(force, settings_overrides=None, selected_files=None, max_workers=1):
            run_calls["selected_files"] = selected_files
            run_calls["max_workers"] = max_workers
            print(f"force={force}")
            print(f"files={selected_files}")
            print(f"workers={max_workers}")
            print(f"settings={settings_overrides['OLLAMA_CHAT_MODEL']}")
            return {"processed": 2, "skipped": 1, "failed": 0, "failures": []}

        with patch.object(server, "_run_ingest", side_effect=fake_run):
            response = self.client.post(
                "/v1/tasks/ingest",
                json={
                    "force": True,
                    "selected_files": ["alpha.pdf", "zeta.pdf"],
                    "max_workers": 4,
                    "settings_overrides": {"OLLAMA_CHAT_MODEL": "qwen-test"},
                },
            )
            task = self.wait_for_task(response.json()["id"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], {"processed": 2, "skipped": 1, "failed": 0, "failures": []})
        self.assertEqual(task["metadata"]["selected_count"], 2)
        self.assertEqual(task["metadata"]["max_workers"], 4)
        self.assertEqual(run_calls["selected_files"], ["alpha.pdf", "zeta.pdf"])
        self.assertEqual(run_calls["max_workers"], 4)
        self.assertIn("force=True", task["log_text"])
        self.assertIn("workers=4", task["log_text"])
        self.assertIn("qwen-test", task["log_text"])

    def test_build_index_task_captures_logs_and_result(self):
        task_started = threading.Event()
        allow_finish = threading.Event()

        def fake_run(recreate, settings_overrides=None, max_workers=1):
            task_started.set()
            if not allow_finish.wait(timeout=2):
                raise RuntimeError("Timed out waiting to finish build-index task.")
            print(f"recreate={recreate}")
            print(f"workers={max_workers}")
            print(f"settings={settings_overrides['OLLAMA_CHAT_MODEL']}")
            return {"indexed": 3}

        with patch.object(server, "_run_build_index", side_effect=fake_run):
            response = self.client.post(
                "/v1/tasks/build-index",
                json={
                    "recreate": True,
                    "max_workers": 4,
                    "settings_overrides": {"OLLAMA_CHAT_MODEL": "qwen-build"},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(task_started.wait(timeout=2))
            self.assertEqual(response.json()["status"], "running")
            allow_finish.set()
            task = self.wait_for_task(response.json()["id"])

        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], {"indexed": 3})
        self.assertEqual(task["metadata"]["max_workers"], 4)
        self.assertIn("recreate=True", task["log_text"])
        self.assertIn("workers=4", task["log_text"])
        self.assertIn("qwen-build", task["log_text"])

    def test_build_index_parallel_document_loading_preserves_order(self):
        with TemporaryDirectory() as temporary_dir:
            processed_dir = Path(temporary_dir)
            first_file = processed_dir / "alpha.json"
            second_file = processed_dir / "zeta.json"
            first_file.write_text(
                '[{"id":"alpha:1","text":"Alpha text","metadata":{"title":"Alpha"}}]',
                encoding="utf-8",
            )
            second_file.write_text(
                '[{"id":"zeta:1","text":"Zeta text","metadata":{"title":"Zeta"}}]',
                encoding="utf-8",
            )
            fake_settings = SimpleNamespace(processed_dir=processed_dir)

            documents = build_index.iter_llama_documents(fake_settings, max_workers=2)

        self.assertEqual([document.doc_id for document in documents], ["alpha:1", "zeta:1"])

    def test_build_index_prints_overall_progress_and_worker_count(self):
        fake_settings = SimpleNamespace(
            processed_dir=Path("/tmp/processed"),
            ollama_embed_model="embed-model",
            ollama_base_url="http://ollama",
            collection_name="articles",
            docstore_path=Path("/tmp/docstore.json"),
            leaf_nodes_path=Path("/tmp/leaf_nodes.json"),
            chunk_sizes=(2048, 768, 256),
            ensure_runtime_dirs=lambda: None,
        )
        fake_document = SimpleNamespace(
            get_metadata_str=lambda mode=None: "metadata",
        )
        fake_node = SimpleNamespace(node_id="leaf-1")
        captured_output = StringIO()

        with (
            patch.object(build_index.RuntimeSettings, "from_env", return_value=fake_settings),
            patch.object(build_index, "ensure_collection", return_value=SimpleNamespace()),
            patch.object(build_index, "SafeOllamaEmbedding", return_value=SimpleNamespace()),
            patch.object(build_index, "iter_llama_documents", return_value=[fake_document]),
            patch.object(build_index, "parse_hierarchical_nodes", return_value=[fake_node]),
            patch.object(build_index, "get_leaf_nodes", return_value=[fake_node]),
            patch.object(build_index, "SimpleDocumentStore") as docstore_class,
            patch.object(build_index, "StorageContext") as storage_context_class,
            patch.object(build_index, "VectorStoreIndex"),
            patch.object(build_index, "persist_leaf_nodes"),
            patch("sys.stdout", captured_output),
        ):
            docstore_class.return_value.add_documents.return_value = None
            docstore_class.return_value.persist.return_value = None
            storage_context_class.from_defaults.return_value = SimpleNamespace()

            result = build_index.build_hierarchical_index(max_workers=4)

        self.assertEqual(result, (1, 1))
        output = captured_output.getvalue()
        self.assertIn("Requested workers/processes: 4", output)
        self.assertIn("Overall index progress: 0% done, 100% remaining", output)
        self.assertIn("Overall index progress: 45% done, 55% remaining", output)
        self.assertIn("Overall index progress: 95% done, 5% remaining", output)
        self.assertIn("Overall index progress: 100% done, 0% remaining", output)

    def test_synthesis_task_captures_logs_and_result(self):
        task_started = threading.Event()
        allow_finish = threading.Event()

        def fake_run(query, verbose, settings_overrides=None):
            task_started.set()
            if not allow_finish.wait(timeout=2):
                raise RuntimeError("Timed out waiting to finish synthesis task.")
            print(f"query={query}")
            print(f"verbose={verbose}")
            return "synthesis-complete"

        with patch.object(server, "_run_synthesis", side_effect=fake_run):
            response = self.client.post(
                "/v1/tasks/synthesis",
                json={
                    "query": "cancer treatment advances",
                    "verbose": True,
                    "settings_overrides": {"OLLAMA_CHAT_MODEL": "qwen-synth"},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(task_started.wait(timeout=2))
            self.assertEqual(response.json()["status"], "running")
            allow_finish.set()
            task = self.wait_for_task(response.json()["id"])

        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"], "synthesis-complete")
        self.assertIn("query=cancer treatment advances", task["log_text"])
        self.assertIn("verbose=True", task["log_text"])

    def test_synthesis_prints_progress_updates_while_waiting_for_model(self):
        query_started = threading.Event()
        allow_finish = threading.Event()
        result_holder = {}
        captured_output = StringIO()

        fake_settings = SimpleNamespace(
            ollama_chat_model="qwen-synth",
            ollama_base_url="http://ollama",
            create_ollama_client=lambda timeout=None: SimpleNamespace(),
            create_ollama_async_client=lambda timeout=None: SimpleNamespace(),
        )

        class FakeMapReduceEngine:
            def query(self, broad_query):
                query_started.set()
                if not allow_finish.wait(timeout=2):
                    raise RuntimeError("Timed out waiting to finish synthesis query.")
                return f"final report for {broad_query}"

        def run_synthesis():
            result_holder["value"] = synthesis.run_global_literature_review(
                "novel therapies",
                status_interval_seconds=0.01,
            )

        with (
            patch.object(synthesis.RuntimeSettings, "from_env", return_value=fake_settings),
            patch.object(synthesis, "get_engine", return_value=SimpleNamespace()),
            patch.object(synthesis, "Ollama", return_value=SimpleNamespace()),
            patch.object(synthesis.SubQuestionQueryEngine, "from_defaults", return_value=FakeMapReduceEngine()),
            patch("sys.stdout", captured_output),
        ):
            worker = threading.Thread(target=run_synthesis)
            worker.start()
            self.assertTrue(query_started.wait(timeout=2))
            time.sleep(0.05)
            allow_finish.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result_holder["value"], "final report for novel therapies")
        output = captured_output.getvalue()
        self.assertIn("Loading runtime settings...", output)
        self.assertIn("Loading query engine and retrieval stack...", output)
        self.assertIn("Submitting the synthesis query to the model...", output)
        self.assertIn("Synthesis is still running", output)
        self.assertIn("Synthesis query completed. Rendering final report...", output)

    def test_synthesis_prints_progress_updates_while_loading_query_engine(self):
        query_engine_started = threading.Event()
        allow_query_engine = threading.Event()
        result_holder = {}
        captured_output = StringIO()

        fake_settings = SimpleNamespace(
            ollama_chat_model="qwen-synth",
            ollama_base_url="http://ollama",
            create_ollama_client=lambda timeout=None: SimpleNamespace(),
            create_ollama_async_client=lambda timeout=None: SimpleNamespace(),
        )

        class FakeMapReduceEngine:
            def query(self, broad_query):
                return f"final report for {broad_query}"

        def fake_get_engine(settings_overrides=None):
            query_engine_started.set()
            if not allow_query_engine.wait(timeout=2):
                raise RuntimeError("Timed out waiting to finish query engine loading.")
            return SimpleNamespace()

        def run_synthesis():
            result_holder["value"] = synthesis.run_global_literature_review(
                "future consciousness",
                status_interval_seconds=0.01,
            )

        with (
            patch.object(synthesis.RuntimeSettings, "from_env", return_value=fake_settings),
            patch.object(
                synthesis,
                "get_engine",
                side_effect=fake_get_engine,
            ),
            patch.object(synthesis, "Ollama", return_value=SimpleNamespace()),
            patch.object(synthesis.SubQuestionQueryEngine, "from_defaults", return_value=FakeMapReduceEngine()),
            patch("sys.stdout", captured_output),
        ):
            worker = threading.Thread(target=run_synthesis)
            worker.start()
            self.assertTrue(query_engine_started.wait(timeout=2))
            time.sleep(0.05)
            allow_query_engine.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result_holder["value"], "final report for future consciousness")
        output = captured_output.getvalue()
        self.assertIn("Loading query engine and retrieval stack...", output)
        self.assertIn("Query engine and retrieval stack are still loading", output)
        self.assertIn("Connecting to the Ollama chat model...", output)
        self.assertIn("Synthesis query completed. Rendering final report...", output)

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


class IngestTests(unittest.TestCase):
    def make_settings(self, temporary_dir: str):
        base_dir = Path(temporary_dir)
        pdf_dir = base_dir / "pdfs"
        processed_dir = base_dir / "processed"
        pdf_dir.mkdir()
        processed_dir.mkdir()
        return SimpleNamespace(
            pdf_dir=pdf_dir,
            processed_dir=processed_dir,
            ensure_runtime_dirs=lambda: None,
        )

    def test_process_corpus_limits_work_to_selected_files(self):
        with TemporaryDirectory() as temporary_dir:
            settings = self.make_settings(temporary_dir)
            (settings.pdf_dir / "alpha.pdf").write_text("", encoding="utf-8")
            (settings.pdf_dir / "zeta.pdf").write_text("", encoding="utf-8")
            processed = []

            def fake_process_pdf(pdf_path, runtime_settings):
                processed.append(pdf_path.name)
                (runtime_settings.processed_dir / f"{pdf_path.stem}.json").write_text("[]", encoding="utf-8")

            with patch.object(ingest.RuntimeSettings, "from_env", return_value=settings), patch.object(
                ingest,
                "process_pdf",
                side_effect=fake_process_pdf,
            ):
                summary = ingest.process_corpus(selected_files=["zeta.pdf"])

        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(processed, ["zeta.pdf"])

    def test_process_corpus_skips_existing_outputs_without_force(self):
        with TemporaryDirectory() as temporary_dir:
            settings = self.make_settings(temporary_dir)
            (settings.pdf_dir / "alpha.pdf").write_text("", encoding="utf-8")
            (settings.processed_dir / "alpha.json").write_text("[]", encoding="utf-8")

            with patch.object(ingest.RuntimeSettings, "from_env", return_value=settings), patch.object(
                ingest,
                "process_pdf",
            ) as process_pdf:
                summary = ingest.process_corpus()

        process_pdf.assert_not_called()
        self.assertEqual(summary["processed"], 0)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["failed"], 0)

    def test_process_corpus_continues_after_file_failure(self):
        with TemporaryDirectory() as temporary_dir:
            settings = self.make_settings(temporary_dir)
            (settings.pdf_dir / "alpha.pdf").write_text("", encoding="utf-8")
            (settings.pdf_dir / "broken.pdf").write_text("", encoding="utf-8")

            def fake_process_pdf(pdf_path, runtime_settings):
                if pdf_path.name == "broken.pdf":
                    raise RuntimeError("cannot parse")
                (runtime_settings.processed_dir / f"{pdf_path.stem}.json").write_text("[]", encoding="utf-8")

            with patch.object(ingest.RuntimeSettings, "from_env", return_value=settings), patch.object(
                ingest,
                "process_pdf",
                side_effect=fake_process_pdf,
            ):
                summary = ingest.process_corpus(max_workers=2)

        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failures"][0]["filename"], "broken.pdf")
        self.assertIn("cannot parse", summary["failures"][0]["error"])

    def test_process_corpus_rejects_unsafe_selected_filename(self):
        with TemporaryDirectory() as temporary_dir:
            settings = self.make_settings(temporary_dir)
            with patch.object(ingest.RuntimeSettings, "from_env", return_value=settings):
                with self.assertRaises(ValueError):
                    ingest.process_corpus(selected_files=["../outside.pdf"])


if __name__ == "__main__":
    unittest.main()

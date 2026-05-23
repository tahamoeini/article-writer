import argparse
import time
from threading import Event, Thread
from typing import Mapping

from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from ollama import ResponseError

try:
    from scripts.config import RuntimeSettings
    from scripts.query_engine import get_engine
except ModuleNotFoundError:
    from config import RuntimeSettings
    from query_engine import get_engine


HEARTBEAT_JOIN_TIMEOUT_SECONDS = 0.2


def _is_ollama_memory_error(exc: BaseException) -> bool:
    return isinstance(exc, ResponseError) and "requires more system memory" in str(exc).lower()


def _build_sub_question_engine(base_engine, llm: Ollama, verbose: bool):
    corpus_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="academic_corpus_tool",
            description=(
                "Provides access to the indexed academic corpus and returns citation-bounded answers."
            ),
        ),
    )
    question_generator = LLMQuestionGenerator.from_defaults(llm=llm)

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[corpus_tool],
        llm=llm,
        question_gen=question_generator,
        verbose=verbose,
    )


def _run_with_periodic_status(
    action,
    *,
    waiting_message: str,
    status_interval_seconds: float,
):
    if status_interval_seconds <= 0:
        return action()

    result_holder = {}
    error_holder = {}
    completed = Event()

    def run_action() -> None:
        try:
            result_holder["value"] = action()
        except BaseException as exc:
            error_holder["error"] = exc
            error_holder["traceback"] = exc.__traceback__
        finally:
            completed.set()

    worker = Thread(target=run_action, daemon=True)
    worker.start()
    started_at = time.monotonic()

    while not completed.wait(status_interval_seconds):
        elapsed_seconds = int(time.monotonic() - started_at)
        print(f"{waiting_message} ({elapsed_seconds}s elapsed)...")

    worker.join(timeout=HEARTBEAT_JOIN_TIMEOUT_SECONDS)

    if "error" in error_holder:
        raise error_holder["error"].with_traceback(error_holder["traceback"])

    return result_holder["value"]


def run_global_literature_review(
    broad_query: str,
    verbose: bool = False,
    settings_overrides: Mapping[str, str] | None = None,
    cancel_event: Event | None = None,
    status_interval_seconds: float = 5.0,
):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Synthesis task was cancelled before it started.")

    print(f"Starting global synthesis workflow for: {broad_query}")
    print("Loading runtime settings...")
    settings = RuntimeSettings.from_env(settings_overrides)
    print(f"Using chat model: {settings.ollama_chat_model}")
    print("Loading query engine and retrieval stack...")
    base_engine = _run_with_periodic_status(
        lambda: get_engine(settings_overrides=settings_overrides),
        waiting_message="Query engine and retrieval stack are still loading",
        status_interval_seconds=status_interval_seconds,
    )
    print("Connecting to the Ollama chat model...")
    llm = Ollama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        request_timeout=600.0,
        client=settings.create_ollama_client(timeout=600.0),
        async_client=settings.create_ollama_async_client(timeout=600.0),
    )

    print("Preparing the sub-question synthesis workflow...")
    map_reduce_engine = _build_sub_question_engine(base_engine, llm, verbose)

    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Synthesis task was cancelled before querying the model.")

    print("Submitting the synthesis query to the model...")
    heartbeat_stop = Event()

    def emit_periodic_progress_updates() -> None:
        started_at = time.monotonic()
        while not heartbeat_stop.wait(status_interval_seconds):
            elapsed_seconds = int(time.monotonic() - started_at)
            print(
                "Synthesis is still running "
                f"({elapsed_seconds}s elapsed). The model may be loading, retrieving evidence, "
                "or drafting the final answer..."
            )

    heartbeat_thread = None
    if status_interval_seconds > 0:
        heartbeat_thread = Thread(target=emit_periodic_progress_updates, daemon=True)
        heartbeat_thread.start()

    try:
        try:
            final_analysis = map_reduce_engine.query(broad_query)
        except ResponseError as exc:
            if not _is_ollama_memory_error(exc):
                raise

            fallback_model = settings.synthesis_fallback_model.strip()
            if not fallback_model or fallback_model == settings.ollama_chat_model:
                raise RuntimeError(
                    "The selected synthesis model needs more memory than currently available. "
                    "Set OLLAMA_CHAT_MODEL to a smaller model, or configure "
                    "OLLAMA_SYNTHESIS_FALLBACK_MODEL to a smaller alternative."
                ) from exc

            print(
                "Primary synthesis model failed due to insufficient memory. "
                f"Retrying with fallback model: {fallback_model}"
            )
            fallback_llm = Ollama(
                model=fallback_model,
                base_url=settings.ollama_base_url,
                request_timeout=600.0,
                client=settings.create_ollama_client(timeout=600.0),
                async_client=settings.create_ollama_async_client(timeout=600.0),
            )
            fallback_engine = _build_sub_question_engine(base_engine, fallback_llm, verbose)
            final_analysis = fallback_engine.query(broad_query)
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=HEARTBEAT_JOIN_TIMEOUT_SECONDS)

    print("Synthesis query completed. Rendering final report...")
    print("\n======================= FINAL REPORT =======================\n")
    print(final_analysis)
    return final_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a global literature synthesis query.")
    parser.add_argument("query", help="Broad literature review question to synthesize across the corpus.")
    parser.add_argument("--verbose", action="store_true", help="Show sub-question workflow details.")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_global_literature_review(arguments.query, verbose=arguments.verbose)

import argparse
from typing import Mapping

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama

try:
    from scripts.config import RuntimeSettings
    from scripts.query_engine import get_advanced_query_engine
except ModuleNotFoundError:
    from config import RuntimeSettings
    from query_engine import get_advanced_query_engine


def run_global_literature_review(
    broad_query: str,
    verbose: bool = False,
    settings_overrides: Mapping[str, str] | None = None,
):
    settings = RuntimeSettings.from_env(settings_overrides)
    base_engine = get_advanced_query_engine(settings_overrides=settings_overrides)
    llm = Ollama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        request_timeout=600.0,
        client=settings.create_ollama_client(timeout=600.0),
        async_client=settings.create_ollama_async_client(timeout=600.0),
    )

    corpus_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="academic_corpus_tool",
            description=(
                "Provides access to the indexed academic corpus and returns citation-bounded answers."
            ),
        ),
    )

    map_reduce_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[corpus_tool],
        llm=llm,
        verbose=verbose,
    )

    print(f"Starting global synthesis workflow for: {broad_query}")
    final_analysis = map_reduce_engine.query(broad_query)
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
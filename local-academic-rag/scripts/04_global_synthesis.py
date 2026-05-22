import os
from query_engine import get_advanced_query_engine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama

def run_global_literature_review(broad_query: str):
    base_engine = get_advanced_query_engine()
    llm = Ollama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", request_timeout=600.0)
    
    # Register the basic core index search as a reusable agent tool
    corpus_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="academic_corpus_tool",
            description="Provides granular search access to the local 1,000 document academic repository."
        )
    )
    
    # Instantiate the Map-Reduce engine
    map_reduce_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[corpus_tool],
        llm=llm,
        verbose=True
    )
    
    print(f"Starting global synthesis workflow execution for: '{broad_query}'")
    final_analysis = map_reduce_engine.query(broad_query)
    
    print("\n======================= FINAL SYNTHESIS REPORT =======================")
    print(final_analysis)

if __name__ == "__main__":
    run_global_literature_review(
        "Provide a comprehensive review mapping how the core theories evolved chronologically across the corpus."
    )
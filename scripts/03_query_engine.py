import os
from qdrant_client import QdrantClient
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

SYSTEM_PROMPT = """You are a strict academic literature review research assistant. 
Your sole function is to synthesize and analyze info exclusively from the provided context blocks.

CRITICAL RULES FOR FACTUAL INTEGRITY:
1. Only base assertions on explicitly retrieved context. Do not extrapolate or assume anything.
2. If the context does not contain direct proof to answer the question, state exactly: 
   "The retrieved corpus does not contain sufficient evidence to answer this question."
3. Every factual claim, argument, or statistic must be followed immediately by an inline citation:
   Format: [Author, Year, "Title", p. X, para. Y]
4. If sources contradict each other, explicitly detail the divergence rather than reconciling it artificially.
5. Do not use external or training dataset memory to provide domain knowledge.
"""

def get_advanced_query_engine():
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name="academic_corpus")
    docstore = SimpleDocumentStore.from_persist_path("./index_storage/docstore.json")
    
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    llm = Ollama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", request_timeout=300.0)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # 1. Base Dense Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=30)
    
    # 2. Base Sparse Keyword (BM25) Retriever 
    all_nodes = list(docstore.docs.values())
    leaf_nodes = [n for n in all_nodes if "is_leaf" in n.metadata or len(n.text) < 500]
    bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=30)
    
    # 3. Hybrid Search (Fuse results using Reciprocal Rank Fusion)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=40,
        mode="reciprocal_rerank",
        use_async=True
    )
    
    # 4. Auto-Merging Parent Context Broker
    auto_merge_retriever = AutoMergingRetriever(
        fusion_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.4,
        verbose=True
    )
    
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="tree_summarize", # Iteratively merges contexts to avoid overloading memory
        extra_info={"system_prompt": SYSTEM_PROMPT}
    )
    
    engine = RetrieverQueryEngine(
        retriever=auto_merge_retriever,
        response_synthesizer=synthesizer
    )
    
    return engine

if __name__ == "__main__":
    engine = get_advanced_query_engine()
    # Test execution pass
    res = engine.query("What are the primary indicators or findings discussed regarding your target variable?")
    print("\n--- SYSTEM RESPONSE ---")
    print(res)
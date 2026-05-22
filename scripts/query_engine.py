import os
from pathlib import Path
from qdrant_client import QdrantClient
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

BASE_DIR = Path(__file__).resolve().parent.parent
DOCSTORE_PATH = BASE_DIR / "index_storage" / "docstore.json"

SYSTEM_PROMPT = """You are a strict academic literature review research assistant.
Your sole function is to synthesize and analyze info exclusively from the provided context blocks.
1. Only base assertions on explicitly retrieved context. Do not extrapolate.
2. If context is insufficient, say: "The retrieved corpus does not contain sufficient evidence."
3. Every claim must have an inline citation: [Author, Year, "Title", p. X, para. Y]
"""

def get_advanced_query_engine():
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    client = QdrantClient(host=QDRANT_HOST, port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name="academic_corpus")
    docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
    
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm = Ollama(model="qwen2.5:7b-instruct", base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    vector_retriever = index.as_retriever(similarity_top_k=30)
    all_nodes = list(docstore.docs.values())
    leaf_nodes = [n for n in all_nodes if "is_leaf" in n.metadata or len(n.text) < 500]
    bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=30)
    
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=40,
        mode="reciprocal_rerank",
        use_async=True
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    auto_merge_retriever = AutoMergingRetriever(
        fusion_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.4
    )
    
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="tree_summarize",
        extra_info={"system_prompt": SYSTEM_PROMPT}
    )
    
    return RetrieverQueryEngine(retriever=auto_merge_retriever, response_synthesizer=synthesizer)
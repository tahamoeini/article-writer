import json
from typing import Mapping

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

try:
    from scripts.config import RuntimeSettings
    from scripts.safe_ollama_embedding import SafeOllamaEmbedding
except ModuleNotFoundError:
    from config import RuntimeSettings
    from safe_ollama_embedding import SafeOllamaEmbedding


SYSTEM_PROMPT = """You are a strict academic literature review research assistant.
Use only the retrieved corpus context when answering.
If the context is insufficient, reply exactly: The retrieved corpus does not contain sufficient evidence.
Every substantive claim must include an inline citation in this format: [Author, Year, \"Title\", p. X, para. Y]
Prefer concise synthesis over speculation.
"""


class CitationAwareQueryEngine:
    def __init__(self, query_engine):
        self._query_engine = query_engine

    def query(self, prompt: str):
        scoped_prompt = f"{SYSTEM_PROMPT}\n\nResearch question:\n{prompt.strip()}"
        return self._query_engine.query(scoped_prompt)


def load_docstore(settings: RuntimeSettings) -> SimpleDocumentStore:
    if not settings.docstore_path.exists():
        raise FileNotFoundError(
            f"Missing docstore artifact at {settings.docstore_path}. Run scripts/build_index.py first."
        )
    return SimpleDocumentStore.from_persist_path(str(settings.docstore_path))


def load_leaf_nodes(settings: RuntimeSettings, docstore: SimpleDocumentStore):
    if settings.leaf_nodes_path.exists():
        with settings.leaf_nodes_path.open("r", encoding="utf-8") as handle:
            node_ids = json.load(handle)
        nodes = [docstore.docs[node_id] for node_id in node_ids if node_id in docstore.docs]
        if nodes:
            return nodes

    return [node for node in docstore.docs.values() if len(getattr(node, "text", "")) <= 500]


def get_advanced_query_engine(
    settings_overrides: Mapping[str, str] | None = None,
) -> CitationAwareQueryEngine:
    settings = RuntimeSettings.from_env(settings_overrides)
    docstore = load_docstore(settings)
    client = settings.create_qdrant_client()
    if not client.collection_exists(settings.collection_name):
        raise FileNotFoundError(
            f"Missing Qdrant collection '{settings.collection_name}'. Run scripts/build_index.py first."
        )

    vector_store = QdrantVectorStore(client=client, collection_name=settings.collection_name)
    embed_model = SafeOllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"trust_env": False},
    )
    llm = Ollama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        request_timeout=300.0,
        client=settings.create_ollama_client(timeout=300.0),
        async_client=settings.create_ollama_async_client(timeout=300.0),
    )

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    vector_retriever = index.as_retriever(similarity_top_k=settings.vector_top_k)
    leaf_nodes = load_leaf_nodes(settings, docstore)
    bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=settings.bm25_top_k)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.fused_top_k,
        num_queries=2,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=False,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    auto_merge_retriever = AutoMergingRetriever(
        fusion_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.4,
    )

    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="tree_summarize",
        use_async=True,
    )
    query_engine = RetrieverQueryEngine(
        retriever=auto_merge_retriever,
        response_synthesizer=synthesizer,
    )
    return CitationAwareQueryEngine(query_engine)
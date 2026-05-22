import os
from dataclasses import dataclass
from pathlib import Path


def _clean_url(value: str) -> str:
    return value.rstrip("/")


def _parse_chunk_sizes(raw_value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("CHUNK_SIZES must contain exactly three comma-separated integers.")
    return tuple(int(part) for part in parts)


@dataclass(frozen=True)
class RuntimeSettings:
    base_dir: Path
    pdf_dir: Path
    processed_dir: Path
    index_dir: Path
    docstore_path: Path
    leaf_nodes_path: Path
    collection_name: str
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str | None
    qdrant_timeout: float
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    grobid_base_url: str
    chunk_sizes: tuple[int, int, int]
    vector_top_k: int
    bm25_top_k: int
    fused_top_k: int

    @property
    def grobid_process_url(self) -> str:
        return f"{self.grobid_base_url}/api/processHeaderDocument"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        base_dir = Path(__file__).resolve().parent.parent
        pdf_dir = base_dir / "corpus" / "pdfs"
        processed_dir = base_dir / "corpus" / "processed"
        index_dir = base_dir / "index_storage"

        return cls(
            base_dir=base_dir,
            pdf_dir=pdf_dir,
            processed_dir=processed_dir,
            index_dir=index_dir,
            docstore_path=index_dir / "docstore.json",
            leaf_nodes_path=index_dir / "leaf_nodes.json",
            collection_name=os.getenv("QDRANT_COLLECTION", "academic_corpus"),
            qdrant_host=os.getenv("QDRANT_HOST", "127.0.0.1"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
            ollama_base_url=_clean_url(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct"),
            grobid_base_url=_clean_url(os.getenv("GROBID_URL", "http://127.0.0.1:8070")),
            chunk_sizes=_parse_chunk_sizes(os.getenv("CHUNK_SIZES", "2048,768,256")),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", "24")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "24")),
            fused_top_k=int(os.getenv("FUSED_TOP_K", "16")),
        )

    def ensure_runtime_dirs(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def create_qdrant_client(self):
        from qdrant_client import QdrantClient

        return QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port,
            api_key=self.qdrant_api_key,
            timeout=self.qdrant_timeout,
            trust_env=False,
        )
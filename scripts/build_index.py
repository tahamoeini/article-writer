import argparse
import json
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, VectorParams

try:
    from scripts.config import RuntimeSettings
    from scripts.safe_ollama_embedding import SafeOllamaEmbedding
except ModuleNotFoundError:
    from config import RuntimeSettings
    from safe_ollama_embedding import SafeOllamaEmbedding


PARSER_EXCLUDED_METADATA_KEYS = ["filename", "source_path", "block_type"]


def load_paragraphs(json_file: Path) -> list[dict[str, object]]:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with json_file.open("r", encoding=encoding) as handle:
                return json.load(handle)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {json_file}")


def iter_llama_documents(settings: RuntimeSettings) -> list[Document]:
    processed_files = sorted(settings.processed_dir.glob("*.json"))
    if not processed_files:
        raise FileNotFoundError(
            f"No processed JSON files found in {settings.processed_dir}. Run scripts/ingest.py first."
        )

    documents: list[Document] = []
    for json_file in processed_files:
        paragraphs = load_paragraphs(json_file)
        for paragraph in paragraphs:
            text = str(paragraph.get("text", "")).strip()
            metadata = paragraph.get("metadata") or {}
            if not text:
                continue

            page = paragraph.get("page")
            paragraph_index = paragraph.get("paragraph_index")
            source_id = str(
                paragraph.get("id")
                or f"{json_file.stem}:{page}:{paragraph_index}"
            )
            document = Document(
                doc_id=source_id,
                text=text,
                metadata={
                    "title": metadata.get("title", json_file.stem),
                    "author": metadata.get("author", "Unknown"),
                    "year": metadata.get("year", "Unknown"),
                    "page": page,
                    "paragraph": paragraph_index,
                    "filename": metadata.get("filename", f"{json_file.stem}.pdf"),
                    "source_path": metadata.get("source_path"),
                    "block_type": paragraph.get("block_type", "paragraph"),
                },
            )
            document.excluded_embed_metadata_keys = PARSER_EXCLUDED_METADATA_KEYS
            document.excluded_llm_metadata_keys = PARSER_EXCLUDED_METADATA_KEYS
            documents.append(document)

    if not documents:
        raise ValueError(f"Processed files exist in {settings.processed_dir} but contain no indexable text.")

    return documents


def resolve_chunk_sizes(settings: RuntimeSettings, documents: list[Document]) -> list[int]:
    chunk_sizes = list(settings.chunk_sizes)
    required_leaf_size = max(
        max(
            len(document.get_metadata_str(mode=MetadataMode.EMBED)),
            len(document.get_metadata_str(mode=MetadataMode.LLM)),
        )
        for document in documents
    ) + 32

    if chunk_sizes[-1] < required_leaf_size:
        chunk_sizes[-1] = required_leaf_size

    for index in range(len(chunk_sizes) - 2, -1, -1):
        if chunk_sizes[index] <= chunk_sizes[index + 1]:
            buffer = 512 if index == 0 else 256
            chunk_sizes[index] = chunk_sizes[index + 1] + buffer

    return chunk_sizes


def ensure_collection(settings: RuntimeSettings, recreate: bool) -> QdrantVectorStore:
    client = settings.create_qdrant_client()
    try:
        collection_exists = client.collection_exists(settings.collection_name)
    except ResponseHandlingException as exc:
        raise RuntimeError(
            "Unable to connect to Qdrant at "
            f"{settings.qdrant_host}:{settings.qdrant_port}. "
            "Start the service first, for example with 'docker compose up -d qdrant'."
        ) from exc

    if recreate and collection_exists:
        client.delete_collection(settings.collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    return QdrantVectorStore(client=client, collection_name=settings.collection_name)


def persist_leaf_nodes(leaf_nodes, settings: RuntimeSettings) -> None:
    payload = [node.node_id for node in leaf_nodes]
    with settings.leaf_nodes_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_hierarchical_index(recreate: bool = False) -> tuple[int, int]:
    settings = RuntimeSettings.from_env()
    settings.ensure_runtime_dirs()

    vector_store = ensure_collection(settings, recreate=recreate)
    docstore = SimpleDocumentStore()
    embed_model = SafeOllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"trust_env": False},
    )

    llama_docs = iter_llama_documents(settings)
    chunk_sizes = resolve_chunk_sizes(settings, llama_docs)

    print(
        f"Loaded {len(llama_docs)} documents from {settings.processed_dir}. "
        f"Using hierarchical chunk sizes: {chunk_sizes}"
    )

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    print("Parsing hierarchical nodes...")
    nodes = node_parser.get_nodes_from_documents(llama_docs, show_progress=True)
    leaf_nodes = get_leaf_nodes(nodes)
    docstore.add_documents(nodes)
    print(f"Parsed {len(nodes)} total nodes with {len(leaf_nodes)} leaf nodes.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)

    print(f"Indexing {len(leaf_nodes)} leaf nodes into collection '{settings.collection_name}'...")
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    docstore.persist(str(settings.docstore_path))
    persist_leaf_nodes(leaf_nodes, settings)
    print(
        "Vector indexing completed. "
        f"docstore={settings.docstore_path}, leaf_manifest={settings.leaf_nodes_path}"
    )
    return len(nodes), len(leaf_nodes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the hierarchical retrieval index.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    build_hierarchical_index(recreate=arguments.recreate)
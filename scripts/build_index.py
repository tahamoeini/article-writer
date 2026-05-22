import argparse
import concurrent.futures
import json
import pickle
from pathlib import Path
from threading import Event
from typing import Mapping

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
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
MAX_WORKERS = 32
INDEX_PROGRESS_PHASES = [
    ("Load processed documents", 10),
    ("Parse hierarchical nodes", 35),
    ("Embed and write vectors", 50),
    ("Persist manifests", 5),
]


def print_overall_progress(done_percent: int, message: str) -> None:
    remaining_percent = max(0, 100 - done_percent)
    print(
        f"Overall index progress: {done_percent}% done, "
        f"{remaining_percent}% remaining - {message}",
        flush=True,
    )


def print_phase_progress(phase_index: int, message: str) -> None:
    completed_percent = sum(percent for _, percent in INDEX_PROGRESS_PHASES[:phase_index])
    phase_name, phase_percent = INDEX_PROGRESS_PHASES[phase_index]
    print_overall_progress(
        completed_percent,
        f"phase {phase_index + 1}/{len(INDEX_PROGRESS_PHASES)}: "
        f"{phase_name} ({phase_percent}% of total) - {message}",
    )


def load_paragraphs(json_file: Path) -> list[dict[str, object]]:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with json_file.open("r", encoding=encoding) as handle:
                return json.load(handle)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {json_file}")


def documents_from_processed_file(json_file: Path) -> list[Document]:
    documents: list[Document] = []
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

    return documents


def load_batch_documents(
    processed_files: list[Path], max_workers: int = 1
) -> list[Document]:
    """Load a batch of processed JSON files into Document objects."""
    documents: list[Document] = []
    max_workers = max(1, min(max_workers, MAX_WORKERS, len(processed_files)))
    if max_workers == 1:
        document_groups = [documents_from_processed_file(f) for f in processed_files]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            document_groups = list(executor.map(documents_from_processed_file, processed_files))

    for document_group in document_groups:
        documents.extend(document_group)
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


def chunk_documents(documents: list[Document], chunk_count: int) -> list[list[Document]]:
    chunk_count = max(1, min(chunk_count, len(documents)))
    chunk_size = (len(documents) + chunk_count - 1) // chunk_count
    return [documents[index:index + chunk_size] for index in range(0, len(documents), chunk_size)]


def parse_nodes_for_documents(documents: list[Document], chunk_sizes: list[int]):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    return node_parser.get_nodes_from_documents(documents, show_progress=False)


def parse_hierarchical_nodes(
    documents: list[Document],
    chunk_sizes: list[int],
    max_workers: int = 1,
):
    max_workers = max(1, min(max_workers, MAX_WORKERS, len(documents)))
    if max_workers == 1:
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        return node_parser.get_nodes_from_documents(documents, show_progress=True)

    document_chunks = chunk_documents(documents, max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        node_groups = list(
            executor.map(
                lambda document_chunk: parse_nodes_for_documents(document_chunk, chunk_sizes),
                document_chunks,
            )
        )

    nodes = []
    for node_group in node_groups:
        nodes.extend(node_group)
    return nodes


def _resolve_embedding_dimension(embed_model: SafeOllamaEmbedding) -> int:
    """Dynamically determine the embedding vector dimension by running a test embed."""
    test_vector = embed_model.get_text_embedding("dimension test")
    return len(test_vector)


def ensure_collection(
    settings: RuntimeSettings, embed_model: SafeOllamaEmbedding, recreate: bool
) -> QdrantVectorStore:
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
        vector_dim = _resolve_embedding_dimension(embed_model)
        print(f"Creating Qdrant collection with vector dimension: {vector_dim}")
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    return QdrantVectorStore(client=client, collection_name=settings.collection_name)


def persist_leaf_nodes(leaf_node_ids: list[str], settings: RuntimeSettings) -> None:
    with settings.leaf_nodes_path.open("w", encoding="utf-8") as handle:
        json.dump(leaf_node_ids, handle, indent=2)


def persist_bm25_index(leaf_nodes, settings: RuntimeSettings) -> None:
    """Build and serialize the BM25 index to disk for fast loading at query time."""
    print(f"Building BM25 index from {len(leaf_nodes)} leaf nodes...")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=leaf_nodes, similarity_top_k=settings.bm25_top_k
    )
    with settings.bm25_index_path.open("wb") as handle:
        pickle.dump(bm25_retriever, handle)
    print(f"BM25 index serialized to {settings.bm25_index_path}")


def build_hierarchical_index(
    recreate: bool = False,
    settings_overrides: Mapping[str, str] | None = None,
    cancel_event: Event | None = None,
    max_workers: int = 1,
) -> tuple[int, int]:
    settings = RuntimeSettings.from_env(settings_overrides)
    settings.ensure_runtime_dirs()
    max_workers = max(1, min(max_workers, MAX_WORKERS))

    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Build index task was cancelled before it started.")

    embed_model = SafeOllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"trust_env": False},
    )
    vector_store = ensure_collection(settings, embed_model, recreate=recreate)

    print(
        "Starting hierarchical index build. "
        f"Requested workers/processes: {max_workers}; "
        f"batch size: {settings.ingest_batch_size}; "
        "document preparation and node parsing can use these worker threads; "
        "embedding/vector writes are performed by the single index builder.",
        flush=True,
    )

    # Discover all processed files
    processed_files = sorted(settings.processed_dir.glob("*.json"))
    if not processed_files:
        raise FileNotFoundError(
            f"No processed JSON files found in {settings.processed_dir}. Run scripts/ingest.py first."
        )

    batch_size = settings.ingest_batch_size
    total_files = len(processed_files)
    batches = [
        processed_files[i:i + batch_size]
        for i in range(0, total_files, batch_size)
    ]
    num_batches = len(batches)

    print_phase_progress(0, f"reading processed JSON files from {settings.processed_dir}")
    print(f"Found {total_files} file(s), processing in {num_batches} batch(es) of up to {batch_size}.")

    # Incremental batch processing
    total_nodes = 0
    all_leaf_node_ids: list[str] = []
    chunk_sizes: list[int] | None = None

    # Load or create docstore incrementally
    if not recreate and settings.docstore_path.exists():
        docstore = SimpleDocumentStore.from_persist_path(str(settings.docstore_path))
        print("Loaded existing docstore for incremental append.")
    else:
        docstore = SimpleDocumentStore()

    for batch_idx, batch_files in enumerate(batches):
        if cancel_event and cancel_event.is_set():
            raise InterruptedError(f"Build index task was cancelled during batch {batch_idx + 1}.")

        batch_label = f"batch {batch_idx + 1}/{num_batches}"
        print(f"\n--- {batch_label}: loading {len(batch_files)} file(s) ---")

        # Phase 1: Load documents for this batch
        documents = load_batch_documents(batch_files, max_workers=max_workers)
        if not documents:
            print(f"{batch_label}: no indexable text found, skipping.")
            continue

        # Resolve chunk sizes on first batch (or if not set)
        if chunk_sizes is None:
            chunk_sizes = resolve_chunk_sizes(settings, documents)
            print(f"Using hierarchical chunk sizes: {chunk_sizes}")

        # Phase 2: Parse hierarchical nodes
        print(f"{batch_label}: parsing {len(documents)} document(s)...")
        nodes = parse_hierarchical_nodes(documents, chunk_sizes, max_workers=max_workers)
        leaf_nodes = get_leaf_nodes(nodes)
        print(f"{batch_label}: {len(nodes)} total nodes, {len(leaf_nodes)} leaf nodes.")

        # Add to docstore
        docstore.add_documents(nodes)

        # Phase 3: Embed and upsert vectors
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        print(f"{batch_label}: embedding and indexing {len(leaf_nodes)} leaf node(s)...")
        VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        # Track totals
        total_nodes += len(nodes)
        all_leaf_node_ids.extend(node.node_id for node in leaf_nodes)

        # Flush docstore to disk after each batch to keep RAM low
        docstore.persist(str(settings.docstore_path))
        print(f"{batch_label}: docstore flushed to disk ({len(all_leaf_node_ids)} leaf nodes so far).")

        # Free batch memory
        del documents, nodes, leaf_nodes

    if not all_leaf_node_ids:
        raise ValueError(f"Processed files exist in {settings.processed_dir} but contain no indexable text.")

    # Calculate progress
    total_leaf_nodes = len(all_leaf_node_ids)
    print_overall_progress(95, f"embedded and indexed {total_leaf_nodes} leaf node(s) across {num_batches} batch(es)")

    # Phase 4: Persist final manifests and BM25 index
    print_phase_progress(3, "writing docstore, leaf-node manifest, and BM25 index")

    # Write leaf node manifest
    persist_leaf_nodes(all_leaf_node_ids, settings)

    # Build and persist BM25 index from all leaf nodes
    all_leaf_nodes = [
        docstore.docs[nid] for nid in all_leaf_node_ids if nid in docstore.docs
    ]
    persist_bm25_index(all_leaf_nodes, settings)

    print_overall_progress(100, "index build complete")
    print(
        "Vector indexing completed. "
        f"docstore={settings.docstore_path}, leaf_manifest={settings.leaf_nodes_path}, "
        f"bm25_index={settings.bm25_index_path}"
    )
    return total_nodes, total_leaf_nodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the hierarchical retrieval index.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"Number of processed JSON files to prepare in parallel, up to {MAX_WORKERS}. Embedding remains single-indexed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the number of files to process per batch (default from INGEST_BATCH_SIZE env or 50).",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    overrides = {}
    if arguments.batch_size is not None:
        overrides["INGEST_BATCH_SIZE"] = str(arguments.batch_size)
    build_hierarchical_index(
        recreate=arguments.recreate,
        settings_overrides=overrides or None,
        max_workers=arguments.workers,
    )

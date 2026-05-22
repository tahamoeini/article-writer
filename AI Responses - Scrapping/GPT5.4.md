# Local Literature Review RAG System: Production-Grade Blueprint

---

## 1. Alternative Architectural Software Stacks

### Comparison Matrix

| Dimension | AnythingLLM | RAGFlow | Dify.ai | LlamaIndex + Qdrant (Custom) |
|---|---|---|---|---|
| PDF Layout Awareness | Naive | **Deep (DeepDoc engine)** | Moderate | Configurable |
| Max Retrieval Chunks | ~4 (hard cap) | ~20 configurable | ~20 configurable | **Unlimited** |
| Hybrid Search (BM25 + Vector) | No | Yes | Partial | **Yes** |
| Reranking Support | No | Yes (BGE) | Plugin-based | **Yes (any model)** |
| Graph RAG | No | No | No | **Yes (LlamaIndex PropertyGraph)** |
| Global Synthesis / Agentic | No | No | Basic | **Full** |
| 1,000-doc Scalability | Poor | Good | Good | **Excellent** |
| Self-hosted Docker | Yes | Yes | Yes | Yes |
| Setup Complexity | Low | Low-Medium | Medium | High |

### Recommendation Tier

**Tier 1 (Recommended): RAGFlow + Qdrant hybrid pipeline**
- RAGFlow's `DeepDoc` engine natively handles multi-column PDFs, tables, and figures with layout-aware parsing. Its chunk merging and reranking are built-in. Run it alongside Qdrant for a production vector store.

**Tier 2 (Maximum Control): LlamaIndex (Python) + Qdrant + Ollama**
- Full programmatic control over every pipeline stage. Required for Graph RAG and agentic map-reduce. Higher setup cost, but the only option that fully solves the global synthesis problem.

**Tier 3 (Rapid Iteration): Dify.ai**
- Good middle ground with a visual workflow builder. Supports custom retrieval pipelines and reranking via API. Use if you want a UI without AnythingLLM's limitations but aren't ready for full custom Python.

**Recommendation for your exact use case:** Start with **RAGFlow** for immediate relief from the top-4 cap and layout parsing issues. Migrate to the **LlamaIndex + Qdrant** custom stack for global synthesis and Graph RAG.

---

## 2. Advanced Document Ingestion & Chunking Strategy

### Phase 1: PDF Pre-Processing Pipeline

The goal is structured, semantically-aware text extraction before any chunking occurs.

**Tool Stack (ordered by quality):**

```
PDFs → Marker (primary) → PyMuPDF (fallback) → Unstructured (tables/figures)
```

- **Marker** (`https://github.com/VikParuchuri/marker`): Converts PDFs to clean Markdown preserving headers, tables, and multi-column layout. Best for academic papers. GPU-accelerated.
- **PyMuPDF (fitz)**: Fastest for clean single-column PDFs. Use for bulk processing when Marker is too slow.
- **Unstructured.io**: Best for mixed-layout documents with embedded tables. Its `partition_pdf` with `strategy="hi_res"` uses a vision model to detect table boundaries.

**Preprocessing Script Skeleton:**

```python
# preprocess_corpus.py
import fitz  # PyMuPDF
from pathlib import Path
import subprocess
import json

def extract_metadata(pdf_path: Path) -> dict:
    """Extract title, authors, year from PDF metadata + first page heuristics."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    first_page_text = doc[0].get_text("text")[:2000]
    return {
        "title": meta.get("title", pdf_path.stem),
        "author": meta.get("author", "Unknown"),
        "year": meta.get("creationDate", "")[:4],
        "source_file": str(pdf_path),
        "page_count": len(doc)
    }

def process_with_marker(pdf_path: Path, output_dir: Path) -> Path:
    """Run Marker for layout-aware Markdown extraction."""
    output_path = output_dir / f"{pdf_path.stem}.md"
    subprocess.run([
        "marker_single", str(pdf_path),
        "--output_dir", str(output_dir),
        "--output_format", "markdown"
    ], check=True)
    return output_path
```

### Phase 2: Chunking Strategy — Hierarchical Parent-Child

Do **not** use fixed-size chunking for academic literature. Use **Hierarchical (Parent-Child) Semantic Chunking**:

```
Document
  └── Section (Parent chunk: ~1,500 tokens) ← fed to LLM
        └── Paragraph (Child chunk: ~200 tokens) ← used for retrieval/embedding
```

**Why this works:**
- Small child chunks give precise embedding representations (no topic dilution).
- When a child chunk is retrieved, the full parent section is sent to the LLM for coherent context.
- Prevents the "4 isolated sentences" problem — the LLM always sees a meaningful passage.

**Implementation with LlamaIndex:**

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore

# Define hierarchy: chunk_sizes = [2048, 512, 128]
# 2048-token nodes = parents (sections)
# 512-token nodes = intermediate
# 128-token nodes = leaf nodes (what gets embedded & retrieved)

parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

documents = SimpleDirectoryReader(
    input_dir="./corpus/processed/",
    filename_as_id=True
).load_data()

# Attach pre-extracted metadata to each document
for doc in documents:
    meta = extract_metadata(Path(doc.metadata["file_path"]))
    doc.metadata.update(meta)
    # Critical: ensure metadata persists to child nodes
    doc.excluded_embed_metadata_keys = []
    doc.excluded_llm_metadata_keys = []

nodes = parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

# Store ALL nodes (parent + child) in docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
```

### Semantic Chunking (Alternative for Thematic Papers)

For papers where section boundaries are unclear, use embedding-based semantic chunking:

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text")

splitter = SemanticSplitterNodeParser(
    buffer_size=2,           # merge 2 adjacent sentences for context
    breakpoint_percentile_threshold=85,  # higher = fewer, larger chunks
    embed_model=embed_model
)
```

Use **Hierarchical for structured papers** (with clear Abstract/Introduction/Methods sections) and **Semantic Chunking for survey papers** or gray literature.

---

## 3. Solving the Top-4 Chunks Limitation

### Architecture: Advanced Retrieval Pipeline

```
Query
  │
  ├─ [BM25 Sparse Search] ──────────────────┐
  │                                          ├── Fusion (RRF) → Top-50 Candidates
  └─ [Dense Vector Search (Embeddings)] ────┘
                                             │
                                    [Cross-Encoder Reranker]
                                             │
                                    Top-10 Child Chunks
                                             │
                                    [Parent Retrieval]
                                             │
                                    Top-10 Parent Sections (1,500 tokens each)
                                             │
                                    [LLM Context Window]
```

### Step 1: Hybrid Search with Qdrant

Qdrant natively supports both dense vectors and sparse vectors (BM25) in a single collection.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams
)

client = QdrantClient(host="localhost", port=6333)

# Create collection with BOTH dense + sparse vectors
client.create_collection(
    collection_name="academic_corpus",
    vectors_config={
        "dense": VectorParams(
            size=768,  # nomic-embed-text dimension
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)
```

**BM25 sparse vector generation:**

```python
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def get_sparse_vector(text: str):
    result = list(sparse_model.embed([text]))[0]
    return {"indices": result.indices.tolist(), "values": result.values.tolist()}
```

### Step 2: Reciprocal Rank Fusion

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=25)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=leaf_nodes,
    similarity_top_k=25
)

fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=50,       # cast a wide net
    num_queries=3,             # generate query variants for better recall
    mode="reciprocal_rerank",  # RRF fusion
    use_async=True,
    verbose=True
)
```

### Step 3: Cross-Encoder Reranking

Pull `bge-reranker-large` locally via HuggingFace (no GPU required for inference on reranker):

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large",
    top_n=10  # compress 50 → 10 most relevant chunks
)
```

### Step 4: Auto-Merging (Child → Parent) Retrieval

```python
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage import StorageContext

storage_context = StorageContext.from_defaults(docstore=docstore)

base_retriever = index.as_retriever(similarity_top_k=25)

auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.4,  # if >40% of parent's children are retrieved, return parent
    verbose=True
)
```

### Assembled Query Engine

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

query_engine = RetrieverQueryEngine.from_args(
    retriever=auto_merging_retriever,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window"),
        reranker,
    ],
    response_synthesizer=...,  # see Section 5
    verbose=True
)
```

---

## 4. Overcoming Context Window & Global Synthesis (Graph/Agentic RAG)

### The Core Problem

A query like *"Trace the evolution of institutional theory across all 1,000 papers"* cannot be answered by any single retrieval pass. The answer requires **aggregating evidence across the entire corpus**.

### Solution A: Agentic Map-Reduce (Immediate, Practical)

```
Query
  │
  ├── [Decompose] → Sub-queries: ["institutional theory 2000-2010", "institutional theory 2010-2020", ...]
  │
  ├── [Map] → Run retrieval + local synthesis on each sub-query independently
  │
  └── [Reduce] → Final LLM call aggregates all sub-answers into a coherent synthesis
```

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama

llm = Ollama(model="qwen2.5:7b-instruct", request_timeout=300.0)

# Wrap your corpus as a tool
corpus_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="academic_corpus",
    description="1,000 academic papers on AI, entrepreneurship, and institutional theory"
)

# Sub-question engine automatically decomposes + map-reduces
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[corpus_tool],
    llm=llm,
    verbose=True,
    use_async=True
)

response = sub_question_engine.query(
    "Trace the chronological evolution of institutional theory across all papers in the corpus"
)
```

### Solution B: Microsoft GraphRAG (Global Synthesis at Scale)

GraphRAG builds a **knowledge graph** from the corpus at index time. Nodes are entities (concepts, authors, institutions), edges are relationships. Queries traverse the graph rather than doing vector search.

**Local GraphRAG deployment:**

```bash
pip install graphrag

# Initialize workspace
python -m graphrag init --root ./graphrag_workspace

# Configure settings.yml to point to Ollama
# (see configuration below)

# Run full index pipeline (builds graph from corpus)
python -m graphrag index --root ./graphrag_workspace
```

**`graphrag_workspace/settings.yml` for Ollama:**

```yaml
llm:
  api_base: http://localhost:11434/v1
  api_key: "ollama"  # placeholder
  model: qwen2.5:7b-instruct
  model_supports_json: true
  max_tokens: 4096
  request_timeout: 300.0
  type: openai_chat

embeddings:
  llm:
    api_base: http://localhost:11434/v1
    api_key: "ollama"
    model: nomic-embed-text
    type: openai_embedding

input:
  type: file
  file_type: text    # use pre-processed .md files from Marker
  base_dir: "./corpus/processed"
  file_pattern: ".*\\.md$"

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  max_gleanings: 1

cluster_graph:
  max_cluster_size: 12

storage:
  type: file
  base_dir: "./graphrag_workspace/output"
```

**Query modes:**

```bash
# LOCAL: specific document/entity queries
python -m graphrag query --root ./graphrag_workspace \
  --method local \
  --query "What does Smith (2023) argue about institutional voids?"

# GLOBAL: synthesis across entire corpus (map-reduce over community summaries)
python -m graphrag query --root ./graphrag_workspace \
  --method global \
  --query "Map the chronological evolution of GenAI entrepreneurship theory"
```

### Solution C: LlamaIndex Property Graph Index (Hybrid)

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor
)

# Build graph at index time
graph_index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    kg_extractors=[
        ImplicitPathExtractor(),       # fast, rule-based
        SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=10)  # LLM-extracted
    ],
    show_progress=True
)

# Query combines graph traversal + vector similarity
graph_query_engine = graph_index.as_query_engine(
    include_text=True,
    retriever_mode="keyword_and_embedding",  # hybrid graph retrieval
    similarity_top_k=20
)
```

---

## 5. Prompt Engineering for Academic Integrity

### System Prompt (Production-Grade)

```
SYSTEM PROMPT — ACADEMIC LITERATURE REVIEW ASSISTANT
=====================================================

You are a strict academic research assistant. Your sole function is to synthesize, 
analyze, and cite information exclusively from the provided document context. 

## ABSOLUTE RULES — NEVER VIOLATE:

1. **SOURCE FIDELITY:** You MUST NOT generate, infer, extrapolate, or fabricate any 
   claim not directly supported by the retrieved context passages. If the context does 
   not contain sufficient information, you MUST state: 
   "The retrieved corpus does not contain sufficient evidence to answer this question."

2. **MANDATORY CITATIONS:** Every factual claim, argument, or paraphrase MUST be 
   followed immediately by an inline citation in this exact format:
   [Author(s) Last Name, Year, "Document Title", p. X] or [Doc Title, §Section Name]
   
   Example: "Institutional voids create substitution markets for formal services 
   [Khanna & Palepu, 2010, "Winning in Emerging Markets", p. 14]."

3. **CITATION ACCURACY:** Only cite sources that appear verbatim in the provided 
   context. Do not cite sources from your training data or general knowledge.

4. **INTELLECTUAL HONESTY:** If sources in the context CONTRADICT each other, you 
   MUST report the contradiction explicitly rather than resolving it artificially.
   Example: "While [Author A, Year] argues X, [Author B, Year] contends the opposite, 
   stating Y."

5. **NO IMPROVISATION:** Do not add commentary, analogies, or elaborations beyond 
   what is directly supported by the provided passages. Academic precision over fluency.

6. **SCOPE DECLARATION:** At the start of your response, state:
   "This response is based on [N] retrieved passages from [M] distinct documents."

## RESPONSE FORMAT:

- Use formal academic English (third person, passive voice where conventional).
- Structure multi-part answers with clear H2/H3 headers.
- Use numbered lists for sequential arguments; bullet points for parallel evidence.
- End every response with a "## References" section listing all cited documents in 
  full bibliographic format (APA 7th edition where metadata is available).

## CONTEXT QUALITY HANDLING:

- If retrieved context is clearly irrelevant to the query, state this explicitly before 
  attempting an answer.
- If only partial information is available, answer only what is supported and explicitly 
  flag what remains unanswered.

You are now ready. Wait for the user's research query.
```

### Query-Side Prompt Template

```python
QUERY_TEMPLATE = """
## Research Query
{query}

## Retrieved Context Passages
The following {n_passages} passages were retrieved from the academic corpus. 
Each passage is tagged with its source metadata.

---
{context_passages}
---

## Instructions
Based ONLY on the passages above, provide a comprehensive academic analysis. 
Apply all rules from your system prompt. Begin your response with the scope declaration.
"""
```

---

## 6. Step-by-Step Deployment Blueprint

### Phase 0: Prerequisites

```yaml
# docker-compose.infrastructure.yml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"   # gRPC port for high-throughput ingestion
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  ragflow:
    image: infiniflow/ragflow:latest
    container_name: ragflow
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ragflow_data:/ragflow/data
      - ./corpus/pdfs:/ragflow/uploads   # mount your PDF folder
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    depends_on:
      - ragflow_mysql
      - ragflow_redis
      - ragflow_es
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

  ragflow_mysql:
    image: mysql:8.0
    container_name: ragflow_mysql
    environment:
      MYSQL_ROOT_PASSWORD: ragflow_root
      MYSQL_DATABASE: ragflow
      MYSQL_USER: ragflow
      MYSQL_PASSWORD: ragflow_pass
    volumes:
      - ragflow_mysql:/var/lib/mysql
    restart: unless-stopped

  ragflow_redis:
    image: redis:7-alpine
    container_name: ragflow_redis
    volumes:
      - ragflow_redis:/data
    restart: unless-stopped

  ragflow_es:
    image: elasticsearch:8.11.0
    container_name: ragflow_es
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - ragflow_es:/usr/share/elasticsearch/data
    restart: unless-stopped

volumes:
  qdrant_storage:
  ragflow_data:
  ragflow_mysql:
  ragflow_redis:
  ragflow_es:
```

### Phase 1: Document Preprocessing (Run Once)

```bash
# 1. Install Marker
pip install marker-pdf fastembed llama-index-core \
    llama-index-llms-ollama llama-index-embeddings-ollama \
    llama-index-retrievers-bm25 llama-index-postprocessor-flag-embedding-reranker \
    qdrant-client sentence-transformers

# 2. Process all PDFs with Marker
marker_chunk_convert ./corpus/pdfs ./corpus/processed \
    --workers 4 \
    --min_length 100

# 3. Verify output - should be .md files for each paper
ls ./corpus/processed/ | wc -l
```

### Phase 2: Metadata Extraction & Enrichment

```python
# scripts/01_extract_metadata.py
import fitz
import json
from pathlib import Path
import re

def extract_structured_metadata(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    first_page = doc[0].get_text("text")
    
    # Heuristic year extraction from first page
    year_match = re.search(r'\b(19|20)\d{2}\b', first_page[:1000])
    
    # Try to extract DOI
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', first_page, re.IGNORECASE)
    
    return {
        "title": meta.get("title") or Path(pdf_path).stem,
        "author": meta.get("author", "Unknown Author"),
        "year": year_match.group(0) if year_match else meta.get("creationDate", "")[:4],
        "doi": doi_match.group(0) if doi_match else None,
        "page_count": len(doc),
        "source_file": str(pdf_path),
        "md_file": str(Path("./corpus/processed") / f"{pdf_path.stem}.md")
    }

# Process all PDFs
pdf_dir = Path("./corpus/pdfs")
metadata_registry = {}

for pdf in pdf_dir.glob("*.pdf"):
    meta = extract_structured_metadata(pdf)
    metadata_registry[pdf.stem] = meta
    print(f"Processed: {meta['title'][:60]} ({meta['year']})")

with open("./corpus/metadata_registry.json", "w") as f:
    json.dump(metadata_registry, f, indent=2)

print(f"\nExtracted metadata for {len(metadata_registry)} documents")
```

### Phase 3: Build the Vector Index

```python
# scripts/02_build_index.py
import json
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import qdrant_client

# Load metadata registry
with open("./corpus/metadata_registry.json") as f:
    metadata_registry = json.load(f)

# Models
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    embed_batch_size=10
)
llm = Ollama(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    request_timeout=600.0,
    context_window=32768
)

# Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Create collection
if not client.collection_exists("academic_corpus"):
    client.create_collection(
        collection_name="academic_corpus",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="academic_corpus"
)

# Load processed Markdown files
print("Loading documents...")
documents = SimpleDirectoryReader(
    input_dir="./corpus/processed",
    required_exts=[".md"],
    filename_as_id=True
).load_data()

# Enrich with metadata
for doc in documents:
    stem = Path(doc.metadata.get("file_path", "")).stem
    if stem in metadata_registry:
        doc.metadata.update(metadata_registry[stem])
    # Critical: pass ALL metadata fields to child nodes
    doc.excluded_embed_metadata_keys = ["md_file", "source_file"]
    doc.excluded_llm_metadata_keys = []

print(f"Loaded {len(documents)} documents")

# Parse into hierarchical nodes
print("Parsing into hierarchical nodes...")
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = parser.get_nodes_from_documents(documents, show_progress=True)
leaf_nodes = get_leaf_nodes(nodes)
print(f"Total nodes: {len(nodes)} | Leaf nodes: {len(leaf_nodes)}")

# Build storage context
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore
)

# Build index (embeds only leaf nodes)
print("Building vector index (this will take a while for 1,000 docs)...")
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

# Persist docstore locally (vector store persists in Qdrant)
storage_context.docstore.persist("./index_storage/docstore.json")
index.storage_context.index_store.persist("./index_storage/index_store.json")

print("Index built and persisted successfully.")
```

### Phase 4: Assemble the Full Query Pipeline

```python
# scripts/03_query_engine.py
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient

SYSTEM_PROMPT = """You are a strict academic research assistant...
[paste full system prompt from Section 5]
"""

def build_query_engine(llm, embed_model):
    # Reload index from storage
    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name="academic_corpus")
    docstore = SimpleDocumentStore.from_persist_path("./index_storage/docstore.json")
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )
    
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    
    # Retrieve all leaf nodes for BM25
    all_nodes = list(docstore.docs.values())
    leaf_nodes = [n for n in all_nodes if n.metadata.get("is_leaf", True)]
    
    # Retrievers
    vector_retriever = index.as_retriever(similarity_top_k=25)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=leaf_nodes, similarity_top_k=25
    )
    
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=50,
        num_queries=3,
        mode="reciprocal_rerank",
        use_async=True
    )
    
    auto_merge_retriever = AutoMergingRetriever(
        fusion_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.4,
        verbose=True
    )
    
    # Reranker
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=10
    )
    
    # Response synthesizer with citation enforcement
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="tree_summarize",  # hierarchical summarization for long contexts
        verbose=True
    )
    
    return RetrieverQueryEngine(
        retriever=auto_merge_retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            reranker
        ]
    )
```

### Phase 5: Interactive CLI + API Server

```python
# scripts/04_serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from query_engine import build_query_engine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

app = FastAPI(title="Literature Review RAG API")

llm = Ollama(model="qwen2.5:7b-instruct", request_timeout=600.0, context_window=32768)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
engine = build_query_engine(llm, embed_model)

class Query(BaseModel):
    question: str
    mode: str = "local"  # "local" | "global" | "agentic"

@app.post("/query")
async def query_corpus(q: Query):
    if q.mode == "local":
        response = engine.query(q.question)
        return {
            "answer": str(response),
            "sources": [
                {
                    "title": n.metadata.get("title"),
                    "author": n.metadata.get("author"),
                    "year": n.metadata.get("year"),
                    "score": n.score,
                    "text_snippet": n.text[:300]
                }
                for n in response.source_nodes
            ]
        }
    # GraphRAG global mode delegated to graphrag CLI subprocess
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 6: Docker Compose — Full Stack

```yaml
# docker-compose.full-stack.yml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

  rag-api:
    build:
      context: ./rag-api
      dockerfile: Dockerfile
    ports: ["8000:8000"]
    volumes:
      - ./corpus:/app/corpus
      - ./index_storage:/app/index_storage
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:8080"]
    environment:
      - OPENAI_API_BASE_URL=http://rag-api:8000/v1  # point UI to your RAG API
      - WEBUI_SECRET_KEY=your-secret-key-here
    depends_on:
      - rag-api
    restart: unless-stopped

volumes:
  qdrant_storage:
```

### Deployment Execution Order

```bash
# 1. Start infrastructure
docker compose -f docker-compose.full-stack.yml up -d qdrant

# 2. Pull required Ollama models
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull bge-reranker-large  # or use HuggingFace directly

# 3. Run preprocessing (one-time, ~2-4 hours for 1,000 PDFs with Marker)
python scripts/01_extract_metadata.py
# Run Marker batch processing
python scripts/02_build_index.py  # ~30-90 min depending on hardware

# 4. Start the API server
docker compose -f docker-compose.full-stack.yml up -d rag-api

# 5. (Optional) Build GraphRAG index for global synthesis
python -m graphrag index --root ./graphrag_workspace
# ~4-8 hours for 1,000 docs — runs once, cached thereafter

# 6. Verify
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main theories of institutional voids?", "mode": "local"}'
```

---

## Summary: Migration Path from AnythingLLM

| Step | Action | Solves |
|---|---|---|
| 1 | Deploy Qdrant + RAGFlow via Docker | Replaces pgvector, gets layout-aware PDF parsing |
| 2 | Run Marker on all 1,000 PDFs | Fixes multi-column/table parsing failures |
| 3 | Build hierarchical nodes (2048/512/128) | Fixes semantic incoherence from flat chunking |
| 4 | Configure BM25 + dense vector hybrid | Improves recall for keyword-heavy academic queries |
| 5 | Add BGE reranker (top-50 → top-10) | Eliminates the top-4 limitation entirely |
| 6 | Enable Auto-Merging retrieval | LLM gets 10 full sections instead of 4 fragments |
| 7 | Apply system prompt from Section 5 | Enforces citation integrity and anti-hallucination |
| 8 | (Advanced) Build GraphRAG index | Enables global corpus synthesis queries |

The most impactful immediate win is **steps 3–6** (hierarchical chunking + hybrid search + reranker + auto-merging). These can be implemented with the Python stack against your existing PDFs before committing to full GraphRAG indexing.

Optimized tool selectionI’m adjusting the earlier architecture to a strict `Ollama + local Python/free tools` constraint and removing every dependency that assumed Hugging Face model downloads.

Under that constraint, the design should change in one important way: remove any dependency on external reranker/model hubs and make Ollama the only model runtime. That rules out the earlier Hugging Face reranker path, but it does not block a strong academic RAG system. The best fit is a custom Python pipeline with Qdrant, hybrid retrieval, parent-child chunking, and an Ollama-based second-pass reranker/synthesizer.

## 1. Best stack choices under `Ollama-only`

1. **Recommended:** `LlamaIndex + Qdrant + Ollama + GROBID/PyMuPDF + FastAPI`
   Best for 1,000 papers, deterministic citations, global synthesis, and no UI-imposed retrieval caps. This is the stack I would use.

2. **Visual workflow option:** `Flowise + Qdrant + Ollama`
   Good if you want a canvas UI, but still expect to push the hard parts into custom Python endpoints. Better than AnythingLLM, but not as clean for large-scale literature review logic.

3. **Optional UI only:** `Open WebUI + your own RAG API`
   Use Open WebUI as the front end and keep all ingestion/retrieval/citation logic in your own FastAPI service. This avoids low-end “chat UI decides retrieval” problems.

**What to avoid under your constraint**
- Any stack whose best feature depends on downloading external reranker or layout models from Hugging Face.
- Anything that hides retrieval settings behind a fixed top-k UI.
- Chroma as the primary store for this workload. It is fine for prototypes, but Qdrant is the better production choice here.

## 2. Revised architecture without Hugging Face

### A. Ingestion and parsing

Use a parsing pipeline that does not rely on HF-hosted models:

1. `GROBID` via Docker for structured academic PDF parsing.
   It is strong on title, authors, references, sections, and scholarly layout.
2. `PyMuPDF` as a fallback for plain text extraction and page anchoring.
3. `pdfplumber` or `Camelot` for tables when needed.
4. `Tesseract OCR` only for scanned PDFs.

**Why GROBID first**
It gives you TEI/XML with section structure and metadata, which is much better than raw text for literature review indexing.

**Metadata to store per chunk**
- `document_title`
- `authors`
- `year`
- `page_start`
- `page_end`
- `section_heading`
- `paragraph_id`
- `source_file`
- `chunk_type` (`child` or `parent`)

That metadata is what makes deterministic citation possible.

### B. Chunking strategy

Do not use flat 500-token chunks. Use hierarchical parent-child chunking:

1. Parent chunks: section-level or subsection-level, about `1200-1800` tokens.
2. Child chunks: paragraph windows, about `150-300` tokens, with light overlap.
3. Retrieval happens on child chunks.
4. LLM context receives the matched parent chunks.

This gives high recall and coherent context at the same time.

### C. Retrieval pipeline without an external reranker model

Because you do not have HF access, replace cross-encoder reranking with a three-stage pipeline:

1. **Hybrid recall**
   - Dense search with an Ollama embedding model such as `nomic-embed-text` or `mxbai-embed-large`
   - Sparse search with BM25
   - Fuse results with Reciprocal Rank Fusion

2. **Parent expansion**
   - Retrieve `40-80` child chunks
   - Collapse them into `8-15` parent sections
   - Deduplicate by document/section

3. **Ollama reranking**
   - Use a fast instruct model in Ollama to score the candidate sections against the query
   - Ask for strict JSON output with relevance score and evidence span ids
   - Keep the top `6-10` parent sections

This is slower than a true cross-encoder, but it works locally, is fully offline, and is acceptable for research workflows if you cache rerank results.

**Practical Ollama rerank prompt**
```text
You are a retrieval reranker.

Given:
- a research query
- a list of candidate passages with ids and metadata

Return valid JSON only:
[
  {
    "id": "candidate_id",
    "score": 0-100,
    "reason": "short reason tied strictly to the passage text"
  }
]

Rules:
- Score based only on direct relevance to the query.
- Prefer passages with explicit definitions, findings, dates, variables, or theoretical claims.
- Do not invent information.
- Do not omit highly relevant candidates.
```

### D. Global synthesis without huge context windows

You do not need a giant context window if you use map-reduce or graph-assisted synthesis.

**Best practical option:** agentic map-reduce with Ollama

1. Decompose the broad question into sub-questions.
2. Run retrieval independently for each sub-question.
3. Produce section summaries with citations.
4. Merge those summaries into a final synthesis.
5. Preserve the source list from every intermediate step.

This works well for prompts like:
- “Trace the evolution of institutional voids from 2000 to 2026”
- “Compare how GenAI is framed in entrepreneurship versus humanitarian applications”

**Graph option under Ollama-only**
You can still build a local property graph with Ollama extracting triples in JSON. Use `networkx` or `neo4j` locally. Keep the graph grounded by storing the exact source paragraph id for every extracted edge.

Example extracted edge:
```json
{
  "subject": "institutional voids",
  "predicate": "increase demand for",
  "object": "intermediary entrepreneurship",
  "source_paragraph_id": "doc42_p7_para3"
}
```

That gives you graph-style traversal without any external model dependency.

## 3. Step-by-step deployment blueprint

### Step 1: Infrastructure

Use these containers:
- `qdrant`
- `grobid`
- `open-webui` optional
- your own `rag-api`

Minimal compose outline:

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  grobid:
    image: lfoppiano/grobid:0.8.1
    ports:
      - "8070:8070"

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - QDRANT_HOST=qdrant
      - GROBID_URL=http://grobid:8070
    depends_on:
      - qdrant
      - grobid
    extra_hosts:
      - "host.docker.internal:host-gateway"

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    depends_on:
      - rag-api

volumes:
  qdrant_storage:
```

### Step 2: Ollama models

Pull only Ollama models:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

**Suggested roles**
- `qwen2.5:7b-instruct` for grounded answering and reranking
- `deepseek-r1:8b` for decomposition and synthesis when you want slower, deeper reasoning
- `nomic-embed-text` or `mxbai-embed-large` for embeddings

### Step 3: Parsing pipeline

1. Send PDFs to GROBID.
2. Store TEI/XML plus normalized Markdown or JSON.
3. Add page anchors from PyMuPDF.
4. Write a clean intermediate JSON record per paper.

Recommended normalized schema:
```json
{
  "doc_id": "paper_001",
  "title": "Paper Title",
  "authors": ["Author A", "Author B"],
  "year": 2024,
  "sections": [
    {
      "heading": "Introduction",
      "page_start": 1,
      "page_end": 3,
      "paragraphs": [
        {
          "paragraph_id": "paper_001_intro_01",
          "text": "...."
        }
      ]
    }
  ]
}
```

### Step 4: Indexing

1. Build parent and child nodes from the normalized JSON.
2. Embed only child nodes.
3. Store all metadata in Qdrant payloads.
4. Keep a local docstore for parent-child relationships.

### Step 5: Retrieval logic

Use this order:

1. Query expansion with Ollama
2. BM25 retrieval top `30-40`
3. Dense retrieval top `30-40`
4. RRF fusion to `40-80`
5. Parent merge
6. Ollama rerank to `6-10`
7. Answer generation with strict citation prompt

### Step 6: Answer-generation rules

Your system prompt should be strict enough that the model fails closed:

```text
You are a strict academic literature review assistant.

Rules:
1. Answer only from the provided context.
2. If the answer is not fully supported by the context, say so explicitly.
3. Every factual claim must include an inline citation in this format:
   [Title | Author | Year | p.X | para.Y]
4. If sources disagree, report the disagreement explicitly.
5. Do not use background knowledge.
6. Do not invent page numbers, authors, years, or claims.
7. If retrieval appears insufficient or off-topic, say:
   "The indexed corpus does not provide sufficient evidence for a reliable answer."

Output style:
- Formal academic tone
- Clear synthesis
- Claims grouped by theme when appropriate
- End with a References section listing only cited items
```

### Step 7: Global review mode

For broad literature review questions, switch from normal QA to map-reduce:

1. Ollama decomposes the question into 5-12 sub-questions.
2. Each sub-question runs through the retrieval pipeline.
3. Each sub-answer is generated with citations only.
4. A final synthesis pass merges sub-answers and preserves cited evidence.

That is the practical replacement for trying to stuff 1,000 documents into one context window.

## Bottom line

Under your actual constraint, I would not use RAGFlow as the core architecture unless you verify every model dependency is local and replaceable. The safest production path is:

1. `GROBID + PyMuPDF` for parsing
2. `Qdrant` for storage
3. `LlamaIndex or plain Python` for retrieval orchestration
4. `Ollama` for embeddings, reranking, decomposition, and answering
5. `Open WebUI` only as a front end, not as the retrieval brain
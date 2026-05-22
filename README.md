# Article Writer

This project builds a local academic retrieval and synthesis pipeline.

It does four things:

1. Extracts structured text and metadata from PDFs in corpus/pdfs.
2. Stores cleaned paragraph JSON files in corpus/processed.
3. Builds a hierarchical retrieval index backed by Qdrant and LlamaIndex.
4. Serves a FastAPI control center with a browser UI, research APIs, background task logs, and model chat.

## Project Layout

- scripts/ingest.py: parses PDFs and writes one JSON file per document into corpus/processed.
- scripts/build_index.py: reads processed JSON files and builds the Qdrant index plus local docstore artifacts.
- scripts/query_engine.py: assembles the hybrid retriever and citation-aware query engine.
- scripts/server.py: exposes the query engine through a FastAPI service plus the browser control center.
- scripts/synthesis.py: runs broader synthesis queries across the indexed corpus.
- scripts/config.py: reads runtime settings and creates clients for Qdrant and Ollama.
- scripts/safe_ollama_embedding.py: wraps Ollama embeddings so long texts are handled safely.
- corpus/pdfs/: place source PDF files here.
- corpus/processed/: generated structured JSON output from ingestion.
- index_storage/: generated docstore and leaf-node artifacts used by the query engine.
- ai-responses/: saved example model outputs and prompt experiments.

## Prerequisites

You need the following services available locally:

- Ollama for embeddings and chat generation.
- Qdrant for vector storage.
- GROBID if you want metadata enrichment from PDF headers.

The repository also includes Docker support for Qdrant, GROBID, and the API service.

## Python Setup

Install the Python dependencies before running the scripts or tests:

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt

On Windows (PowerShell):

    .venv\\Scripts\\Activate.ps1

## Environment Variables

The scripts read their settings from environment variables, with these defaults:

- QDRANT_HOST: 127.0.0.1
- QDRANT_PORT: 6333
- QDRANT_COLLECTION: academic_corpus
- QDRANT_API_KEY: empty
- QDRANT_TIMEOUT: 30
- OLLAMA_BASE_URL: http://127.0.0.1:11434
- OLLAMA_EMBED_MODEL: nomic-embed-text
- OLLAMA_CHAT_MODEL: qwen2.5:7b-instruct
- GROBID_URL: http://127.0.0.1:8070
- CHUNK_SIZES: 2048,768,256
- VECTOR_TOP_K: 24
- BM25_TOP_K: 24
- FUSED_TOP_K: 16

The project writes generated artifacts to:

- corpus/processed/
- index_storage/docstore.json
- index_storage/leaf_nodes.json

## Recommended Workflow

### 1. Add PDFs

Copy PDF files into corpus/pdfs.

### 2. Extract structured text

Run the ingestion script to convert PDFs into JSON:

    python scripts/ingest.py

This creates one JSON file per PDF in corpus/processed. Existing outputs are skipped unless you pass --force.

To rebuild all processed files:

    python scripts/ingest.py --force

To process multiple PDFs in parallel:

    python scripts/ingest.py --workers 4

To ingest only selected files already present in corpus/pdfs:

    python scripts/ingest.py --file first.pdf --file second.pdf --workers 4

### 3. Build the index

After processing is complete, build the retrieval index:

    python scripts/build_index.py

If you want to delete and recreate the Qdrant collection first:

    python scripts/build_index.py --recreate

This step:

- loads corpus/processed/*.json
- parses hierarchical nodes
- stores vectors in Qdrant
- writes the local docstore to index_storage/docstore.json
- writes leaf node IDs to index_storage/leaf_nodes.json

### 4. Open the web UI

Start the API server:

    python scripts/server.py

Then open:

    http://127.0.0.1:8000

The web UI lets you:

- run ingestion, index-building, and synthesis jobs from the browser
- select existing PDFs from corpus/pdfs and configure parallel ingestion workers
- inspect live task logs and final task results
- load Ollama models and switch chat / embedding models
- run corpus-grounded research queries
- chat directly with the selected Ollama model

### 5. Query the corpus through the API

The server still listens on port 8000 by default and exposes API endpoints behind the UI.

Health check:

    GET /health

Research query endpoint:

    POST /v1/research/query

Example request body:

    {
      "prompt": "What does the corpus say about hierarchical retrieval methods?"
    }

The response includes:

- text: the generated answer
- citations: source nodes with title, author, year, page, paragraph, and score

Task endpoints exposed for the UI:

- POST /v1/tasks/ingest
- POST /v1/tasks/build-index
- POST /v1/tasks/synthesis
- GET /v1/tasks
- GET /v1/tasks/{task_id}
- GET /v1/corpus/pdfs
- GET /v1/models
- POST /v1/chat
- GET /v1/settings/defaults

Example ingestion task body:

    {
      "force": false,
      "selected_files": ["first.pdf", "second.pdf"],
      "max_workers": 4
    }

If selected_files is empty or omitted, ingestion scans all PDFs in corpus/pdfs. Per-file failures are logged and reported without stopping the rest of the batch.

### 6. Run a broader synthesis query from the CLI

Use the synthesis script when you want a multi-step literature review style answer:

    python scripts/synthesis.py "What are the main themes across the corpus?"

Add --verbose to inspect the sub-question workflow:

    python scripts/synthesis.py "What are the main themes across the corpus?" --verbose

## Docker Setup

The docker-compose file starts three services:

- qdrant on port 6333
- grobid on port 8070
- rag-api on port 8000 for both the UI and the API

Bring the stack up with:

    docker compose up -d --build

The API container expects Ollama to be reachable from the host machine at http://host.docker.internal:11434.

## How the Pieces Fit Together

The pipeline is intentionally split into separate stages:

1. ingest.py extracts text and metadata from PDFs.
2. build_index.py turns the processed JSON into a retrievable corpus.
3. query_engine.py loads the docstore and Qdrant collection and applies hybrid retrieval.
4. server.py exposes the engine as an HTTP API.
5. synthesis.py uses the query engine as a tool for broader corpus-wide synthesis.

This separation lets you rebuild only the stage you changed instead of rerunning everything.

## Troubleshooting

- If ingesting fails, verify that PyMuPDF is installed and that corpus/pdfs contains PDF files.
- If build_index.py reports a missing collection or connection error, start Qdrant first.
- If the UI health check returns a degraded status, check that Qdrant, Ollama, and the docstore artifacts exist.
- If GROBID is unavailable, ingestion still works, but metadata enrichment falls back to local PDF extraction.
- If queries fail with missing artifacts, rebuild the index after ingestion.

## Validation

The current automated server checks can be run with:

    python -m unittest discover -s tests

## Notes on Generated Files

The following paths are generated and can usually be rebuilt from source PDFs:

- corpus/processed/
- index_storage/

The ai-responses/ folder is for saved outputs and prompt experiments. It is safe to keep examples there for reference.

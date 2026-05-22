FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    llama-index-core \
    llama-index-vector-stores-qdrant \
    llama-index-llms-ollama \
    llama-index-embeddings-ollama \
    llama-index-retrievers-bm25 \
    qdrant-client \
    pymupdf \
    fastapi \
    jinja2 \
    uvicorn \
    pydantic \
    requests

COPY . /app

EXPOSE 8000

# Fixes the python module path evaluation error natively
ENV PYTHONPATH=/app

CMD ["python", "scripts/server.py"]
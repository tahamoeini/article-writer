# --- Builder Stage ---
FROM python:3.13-slim AS builder

WORKDIR /app

# Prevent Python from writing pyc files and address buffering for logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install/upgrade pip and python dependencies in virtual environment
RUN pip install --upgrade pip && \
    pip install \
    llama-index-core \
    llama-index-vector-stores-qdrant \
    llama-index-llms-ollama \
    llama-index-embeddings-ollama \
    llama-index-retrievers-bm25 \
    qdrant-client \
    pymupdf \
    fastapi \
    uvicorn \
    pydantic

# --- Runner Stage ---
FROM python:3.13-slim AS runner

WORKDIR /app

# Set execution environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install necessary lightweight tools for health checks/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy codebase into image
COPY . /app

# Use a non-root system user for security hardening
RUN useradd -u 1000 --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API runtime port
EXPOSE 8000

# Run API as entrypoint
CMD ["python", "scripts/05_api_server.py"]

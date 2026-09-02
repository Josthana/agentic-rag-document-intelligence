# =========================================================
# Agentic RAG Document Intelligence API
# =========================================================

FROM python:3.13-slim


# ---------------------------------------------------------
# Python environment
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1


# ---------------------------------------------------------
# Working directory
# ---------------------------------------------------------

WORKDIR /app


# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt


# ---------------------------------------------------------
# Copy application
# ---------------------------------------------------------

COPY app ./app
COPY data ./data
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY tests ./tests

COPY pytest.ini .


# ---------------------------------------------------------
# Create runtime directories
# ---------------------------------------------------------

RUN mkdir -p /app/state/chroma \
    && mkdir -p /app/evaluation/results


# ---------------------------------------------------------
# Default environment configuration
#
# Ollama itself is running on the Mac.
# Docker will access it through host.docker.internal.
# ---------------------------------------------------------

ENV LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV EMBEDDING_DIMENSION=384

ENV CHROMA_DIR=/app/state/chroma
ENV CHROMA_COLLECTION=documents
ENV VECTOR_TOP_K=5

ENV BM25_TOP_K=5
ENV BM25_INDEX_FILE=/app/state/bm25_index.json

ENV FINAL_TOP_K=5
ENV VECTOR_WEIGHT=0.6
ENV BM25_WEIGHT=0.4

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV OLLAMA_TIMEOUT=120


# ---------------------------------------------------------
# API port
# ---------------------------------------------------------

EXPOSE 8000


# ---------------------------------------------------------
# Container health check
# ---------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1


# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
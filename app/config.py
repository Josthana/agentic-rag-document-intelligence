import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


def get_required_env(name: str) -> str:
    """
    Return a required environment variable.

    Raises:
        ValueError: If the variable does not exist.
    """

    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}"
        )

    return value


# =========================================================
# GEMINI CONFIGURATION
# =========================================================
#
# Gemini will be used later during the RAG generation phase.
# It is NOT used by ChromaDB or BM25 retrieval.
#

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.7-flash",
)

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)


# =========================================================
# LEGACY PINECONE CONFIGURATION
# =========================================================
#
# These are temporarily kept because some older project
# files may still reference them.
#

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-rag-docs",
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "documents",
)


# =========================================================
# LOCAL EMBEDDING CONFIGURATION
# =========================================================
#
# SentenceTransformer + Chroma use these settings.
#

EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "384",
    )
)


# =========================================================
# APPLICATION SETTINGS
# =========================================================

@dataclass
class Settings:

    embedding_model: str = EMBEDDING_MODEL
    embedding_dimension: int = EMBEDDING_DIMENSION

    chroma_dir: Path = Path(
        os.getenv(
            "CHROMA_DIR",
            "state/chroma",
        )
    )

    chroma_collection: str = os.getenv(
        "CHROMA_COLLECTION",
        "documents",
    )

    vector_top_k: int = int(
        os.getenv(
            "VECTOR_TOP_K",
            "5",
        )
    )

    bm25_top_k: int = int(
        os.getenv(
            "BM25_TOP_K",
            "5",
        )
    )

    bm25_index_file: Path = Path(
        os.getenv(
            "BM25_INDEX_FILE",
            "state/bm25_index.json",
        )
    )

    final_top_k: int = int(
        os.getenv(
            "FINAL_TOP_K",
            "5",
        )
    )

    vector_weight: float = float(
        os.getenv(
            "VECTOR_WEIGHT",
            "0.6",
        )
    )

    bm25_weight: float = float(
        os.getenv(
            "BM25_WEIGHT",
            "0.4",
        )
    )

    # Ollama
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b",
    )

    ollama_timeout: int = int(
        os.getenv(
            "OLLAMA_TIMEOUT",
            "120",
        )
    )
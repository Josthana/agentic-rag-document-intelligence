from dataclasses import dataclass, field
from typing import Any


# =========================================================
# DOCUMENT CHUNK
# =========================================================

@dataclass
class DocumentChunk:
    """
    A chunk of text created during document ingestion.

    This is the format stored by:
    - Chroma vector retrieval
    - BM25 keyword retrieval
    """

    chunk_id: str

    text: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# RETRIEVED CHUNK
# =========================================================

@dataclass
class RetrievedChunk:
    """
    A document chunk returned by a retrieval system.
    """

    chunk_id: str

    text: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0

    source: str = ""


# =========================================================
# RAG RESULT
# =========================================================

@dataclass
class RAGResult:
    """
    Final output produced by the RAG or
    Agentic RAG pipeline.
    """

    query: str

    answer: str

    sources: list[RetrievedChunk] = field(
        default_factory=list
    )

    used_query: str | None = None
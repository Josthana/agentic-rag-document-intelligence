from dataclasses import dataclass, field
from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from app.config import (
    CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
)


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class DocumentChunk:
    """
    A chunk of text created from an ingested document.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RetrievedChunk:
    """
    A chunk returned by a retrieval system.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    score: float = 0.0
    source: str = ""

@dataclass
class RAGResult:
    """
    Final result returned by the RAG pipeline.
    """

    query: str
    answer: str
    sources: list[RetrievedChunk] = field(
        default_factory=list
    )
    used_query: str | None = None
# =========================================================
# GEMINI LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0,
)


# =========================================================
# LEGACY GEMINI EMBEDDINGS
# =========================================================
#
# Keep these temporarily because some older project files
# may still import them.
#
# The new Chroma vector store does NOT use these.
# It uses SentenceTransformer instead.
#

document_embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL,
    task_type="RETRIEVAL_DOCUMENT",
)


query_embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL,
    task_type="RETRIEVAL_QUERY",
)
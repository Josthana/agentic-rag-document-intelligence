from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings
from app.graph import AgenticRAGService
from app.ingestion import (
    chunk_documents,
    load_pdfs,
    remove_empty_documents,
)
from app.models import DocumentChunk
from app.bm25_store import BM25Store
from app.retriever import HybridRetriever
from app.vector_store import VectorStore


# =========================================================
# API REQUEST / RESPONSE MODELS
# =========================================================


class QueryRequest(BaseModel):
    """
    Request sent by the user to the RAG API.
    """

    query: str = Field(
        ...,
        min_length=2,
        description="Question to ask the document system.",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of retrieved chunks.",
    )


class SourceResponse(BaseModel):
    """
    Source information returned with the answer.
    """

    chunk_id: str
    source: str
    page: int | str | None
    score: float
    retriever: str


class QueryResponse(BaseModel):
    """
    Final API response.
    """

    query: str
    answer: str
    used_query: str | None
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """
    Health-check response.
    """

    status: str
    vector_chunks: int
    bm25_chunks: int
    model: str


# =========================================================
# APPLICATION INITIALIZATION
# =========================================================


def build_agent() -> tuple[
    AgenticRAGService,
    VectorStore,
    BM25Store,
]:
    """
    Build the complete Agentic RAG system.

    Pipeline:

    PDFs
        ↓
    ingestion
        ↓
    chunking
        ↓
    VectorStore + BM25Store
        ↓
    HybridRetriever
        ↓
    AgenticRAGService
    """

    settings = Settings()

    print("\n====================================")
    print("STARTING AGENTIC RAG API")
    print("====================================\n")

    # -----------------------------------------------------
    # 1. Load PDFs
    # -----------------------------------------------------

    print("[API] Loading PDF documents...")

    documents = load_pdfs(
        "data/raw"
    )

    documents = remove_empty_documents(
        documents
    )

    print(
        f"[API] Loaded "
        f"{len(documents)} non-empty page(s)."
    )

    # -----------------------------------------------------
    # 2. Chunk documents
    # -----------------------------------------------------

    print("[API] Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"[API] Created {len(chunks)} chunk(s)."
    )

    # -----------------------------------------------------
    # 3. Convert to DocumentChunk
    # -----------------------------------------------------

    document_chunks: list[DocumentChunk] = []

    for index, chunk in enumerate(chunks):

        chunk_id = chunk.metadata.get(
            "chunk_id",
            f"chunk-{index:05d}",
        )

        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=chunk.page_content,
                metadata=chunk.metadata,
            )
        )

    # -----------------------------------------------------
    # 4. Initialize retrieval stores
    # -----------------------------------------------------

    print("[API] Initializing VectorStore...")

    vector_store = VectorStore(
        settings
    )

    print("[API] Initializing BM25Store...")

    bm25_store = BM25Store(
        settings
    )

    # -----------------------------------------------------
    # 5. Index chunks
    # -----------------------------------------------------

    print("[API] Indexing document chunks...")

    vector_store.add_chunks(
        document_chunks
    )

    bm25_store.add_chunks(
        document_chunks
    )

    print(
        f"[API] Vector chunks: "
        f"{vector_store.count()}"
    )

    print(
        f"[API] BM25 chunks: "
        f"{len(bm25_store.chunks)}"
    )

    # -----------------------------------------------------
    # 6. Hybrid retriever
    # -----------------------------------------------------

    retriever = HybridRetriever(
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # -----------------------------------------------------
    # 7. LangGraph agent
    # -----------------------------------------------------

    agent = AgenticRAGService(
        settings=settings,
        retriever=retriever,
    )

    print("\n[API] Agentic RAG system ready.\n")

    return (
        agent,
        vector_store,
        bm25_store,
    )


# =========================================================
# FASTAPI LIFESPAN
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG system once when FastAPI starts.
    """

    (
        agent,
        vector_store,
        bm25_store,
    ) = build_agent()

    app.state.agent = agent
    app.state.vector_store = vector_store
    app.state.bm25_store = bm25_store
    app.state.settings = Settings()

    yield

    print(
        "\n[API] Shutting down Agentic RAG API."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================


app = FastAPI(
    title="Agentic RAG Document Intelligence API",
    description=(
        "Hybrid retrieval and LangGraph-powered "
        "document question-answering system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# ROUTES
# =========================================================


@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "message": (
            "Agentic RAG Document "
            "Intelligence API"
        ),
        "docs": "/docs",
        "health": "/health",
        "query": "/query",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Verify that the API and retrieval stores
    are ready.
    """

    settings = app.state.settings

    return HealthResponse(
        status="healthy",
        vector_chunks=(
            app.state.vector_store.count()
        ),
        bm25_chunks=len(
            app.state.bm25_store.chunks
        ),
        model=settings.ollama_model,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_documents(
    request: QueryRequest,
):
    """
    Ask a question using the complete
    LangGraph Agentic RAG pipeline.
    """

    try:

        result = app.state.agent.answer(
            request.query
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    sources = []

    for source in result.sources:

        sources.append(
            SourceResponse(
                chunk_id=source.chunk_id,
                source=source.metadata.get(
                    "source",
                    "Unknown",
                ),
                page=source.metadata.get(
                    "page_number",
                    source.metadata.get(
                        "page"
                    ),
                ),
                score=float(
                    source.score
                ),
                retriever=source.source,
            )
        )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        used_query=result.used_query,
        sources=sources,
    )
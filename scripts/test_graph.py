from app.config import Settings

from app.ingestion import (
    load_pdfs,
    remove_empty_documents,
    chunk_documents,
)

from app.models import DocumentChunk

from app.vector_store import VectorStore
from app.bm25_store import BM25Store
from app.retriever import HybridRetriever
from app.graph import AgenticRAGService


def build_agent():
    """
    Build the complete Agentic RAG pipeline.
    """

    settings = Settings()

    print("1. Loading PDF documents...")

    documents = load_pdfs(
        "data/raw"
    )

    documents = remove_empty_documents(
        documents
    )

    print(
        f"Loaded {len(documents)} "
        "non-empty page(s).\n"
    )

    print("2. Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"Created {len(chunks)} chunk(s).\n"
    )

    print(
        "3. Creating DocumentChunk objects..."
    )

    document_chunks = []

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

    print(
        f"Created {len(document_chunks)} "
        "DocumentChunk object(s).\n"
    )

    print(
        "4. Initializing retrieval stores..."
    )

    vector_store = VectorStore(
        settings
    )

    bm25_store = BM25Store(
        settings
    )

    print("5. Indexing chunks...")

    vector_store.add_chunks(
        document_chunks
    )

    bm25_store.add_chunks(
        document_chunks
    )

    print(
        f"Vector chunks: "
        f"{vector_store.count()}"
    )

    print(
        f"BM25 chunks: "
        f"{len(bm25_store.chunks)}\n"
    )

    print(
        "6. Creating HybridRetriever..."
    )

    retriever = HybridRetriever(
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    print(
        "7. Creating AgenticRAGService..."
    )

    agent = AgenticRAGService(
        settings=settings,
        retriever=retriever,
    )

    print(
        "Agentic RAG service ready.\n"
    )

    return agent


def display_result(result):
    """
    Print the final graph result.
    """

    print(
        "\n===================================="
    )

    print("ANSWER")

    print(
        "====================================\n"
    )

    print(
        result.answer
    )

    print(
        "\n===================================="
    )

    print("QUERY INFORMATION")

    print(
        "====================================\n"
    )

    print(
        f"Original query: {result.query}"
    )

    print(
        f"Query used for retrieval: "
        f"{result.used_query}"
    )

    print(
        "\n===================================="
    )

    print("SOURCES")

    print(
        "====================================\n"
    )

    if not result.sources:

        print(
            "No sources returned."
        )

        return

    for number, source in enumerate(
        result.sources,
        start=1,
    ):

        print(
            f"[Source {number}]"
        )

        print(
            "File:",
            source.metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            source.metadata.get(
                "page_number",
                "Unknown",
            ),
        )

        print(
            f"Score: {source.score:.4f}"
        )

        print(
            f"Retriever: {source.source}"
        )

        print(
            f"Chunk ID: {source.chunk_id}"
        )

        print()


def main():

    print(
        "\n===================================="
    )

    print(
        "PHASE 7 - LANGGRAPH AGENT TEST"
    )

    print(
        "====================================\n"
    )

    agent = build_agent()

    # =================================================
    # TEST 1
    #
    # Normal question.
    # Expected route:
    #
    # retrieve
    #    ↓
    # sufficient context
    #    ↓
    # generate
    # =================================================

    query = (
        "What are the requirements "
        "for remote work eligibility?"
    )

    print(
        "\n===================================="
    )

    print("TEST 1 - NORMAL RAG ROUTE")

    print(
        "====================================\n"
    )

    print(
        f"Question:\n{query}\n"
    )

    result = agent.answer(
        query
    )

    display_result(
        result
    )

    # =================================================
    # TEST 2
    #
    # Force the agentic retry path.
    #
    # Vector/hybrid scores are <= 1, so setting
    # the threshold above 1 guarantees that the
    # graph considers the context weak.
    #
    # Expected route:
    #
    # retrieve
    #    ↓
    # weak
    #    ↓
    # rewrite
    #    ↓
    # retrieve again
    #    ↓
    # no_context
    # =================================================

    print(
        "\n===================================="
    )

    print(
        "TEST 2 - QUERY REWRITE ROUTE"
    )

    print(
        "====================================\n"
    )

    original_threshold = (
        agent.minimum_context_score
    )

    agent.minimum_context_score = 1.01

    retry_query = (
        "Tell me the policy about "
        "working away from the office."
    )

    print(
        f"Question:\n{retry_query}\n"
    )

    retry_result = agent.answer(
        retry_query
    )

    display_result(
        retry_result
    )

    # Restore production threshold.
    agent.minimum_context_score = (
        original_threshold
    )

    print(
        "\n===================================="
    )

    print(
        "PHASE 7 LANGGRAPH AGENT TEST COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
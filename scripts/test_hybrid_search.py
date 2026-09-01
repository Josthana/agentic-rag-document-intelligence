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


def main():

    print("\n====================================")
    print("PHASE 5 - HYBRID RETRIEVAL TEST")
    print("====================================\n")

    settings = Settings()

    # -------------------------------------------------
    # 1. Load PDFs
    # -------------------------------------------------

    print("1. Loading PDFs...")

    documents = load_pdfs(
        "data/raw"
    )

    print(
        f"Loaded {len(documents)} page(s).\n"
    )

    # -------------------------------------------------
    # 2. Remove empty pages
    # -------------------------------------------------

    print("2. Removing empty documents...")

    documents = remove_empty_documents(
        documents
    )

    print(
        f"{len(documents)} non-empty page(s).\n"
    )

    # -------------------------------------------------
    # 3. Chunk documents
    # -------------------------------------------------

    print("3. Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"Created {len(chunks)} chunk(s).\n"
    )

    # -------------------------------------------------
    # 4. Convert to DocumentChunk
    # -------------------------------------------------

    print(
        "4. Creating DocumentChunk objects..."
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

    # -------------------------------------------------
    # 5. Initialize stores
    # -------------------------------------------------

    print(
        "5. Initializing retrieval stores..."
    )

    vector_store = VectorStore(
        settings
    )

    bm25_store = BM25Store(
        settings
    )

    # -------------------------------------------------
    # 6. Add chunks to both stores
    # -------------------------------------------------

    print(
        "6. Indexing chunks..."
    )

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

    # -------------------------------------------------
    # 7. Initialize hybrid retriever
    # -------------------------------------------------

    retriever = HybridRetriever(
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # -------------------------------------------------
    # 8. Run hybrid search
    # -------------------------------------------------

    query = (
        "What are the requirements "
        "for remote work eligibility?"
    )

    print(
        "7. Running hybrid retrieval...\n"
    )

    print(
        f"Query: {query}\n"
    )

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    # -------------------------------------------------
    # 9. Display results
    # -------------------------------------------------

    if not results:

        print(
            "No hybrid results found."
        )

        return

    print(
        f"Top {len(results)} "
        "hybrid result(s):\n"
    )

    for number, result in enumerate(
        results,
        start=1,
    ):

        print(
            "------------------------------------"
        )

        print(
            f"RESULT {number}"
        )

        print(
            "------------------------------------"
        )

        print(
            f"Chunk ID: {result.chunk_id}"
        )

        print(
            f"Hybrid score: "
            f"{result.score:.4f}"
        )

        print(
            f"Retriever: {result.source}"
        )

        print(
            "Source:",
            result.metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            result.metadata.get(
                "page_number",
                "Unknown",
            ),
        )

        print("\nText:\n")

        print(
            result.text[:700]
        )

        print()

    print(
        "===================================="
    )

    print(
        "PHASE 5 HYBRID RETRIEVAL TEST COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
    
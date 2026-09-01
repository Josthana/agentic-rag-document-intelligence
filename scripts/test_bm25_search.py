from app.config import Settings

from app.ingestion import (
    load_pdfs,
    remove_empty_documents,
    chunk_documents,
)

from app.models import DocumentChunk
from app.bm25_store import BM25Store


def main():

    print("\n====================================")
    print("PHASE 4 - BM25 SEARCH TEST")
    print("====================================\n")

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
    # 3. Create chunks
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
    # 5. Initialize BM25
    # -------------------------------------------------

    print("5. Initializing BM25Store...")

    store = BM25Store(
        Settings()
    )

    # -------------------------------------------------
    # 6. Add chunks
    # -------------------------------------------------

    print("6. Building BM25 index...")

    store.add_chunks(
        document_chunks
    )

    print(
        f"BM25 index contains "
        f"{len(store.chunks)} chunk(s).\n"
    )

    # -------------------------------------------------
    # 7. Search
    # -------------------------------------------------

    query = (
        "remote work eligibility"
    )

    print("7. Running BM25 search...\n")

    print(
        f"Query: {query}\n"
    )

    results = store.search(
        query=query,
        top_k=3,
    )

    # -------------------------------------------------
    # 8. Display results
    # -------------------------------------------------

    if not results:
        print(
            "No BM25 results found."
        )

        return

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
            f"Score: {result.score:.4f}"
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
        "PHASE 4 BM25 SEARCH TEST COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
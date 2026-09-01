from app.config import Settings
from app.ingestion import (
    load_pdfs,
    remove_empty_documents,
    chunk_documents,
)
from app.models import DocumentChunk
from app.vector_store import VectorStore


def main():
    print("\n====================================")
    print("PHASE 3 - VECTOR SEARCH TEST")
    print("====================================\n")

    # -----------------------------------------------------
    # 1. Load PDF files
    # -----------------------------------------------------

    print("1. Loading PDF documents...")

    documents = load_pdfs(
        "data/raw"
    )

    print(
        f"Loaded {len(documents)} page(s).\n"
    )

    # -----------------------------------------------------
    # 2. Remove empty pages
    # -----------------------------------------------------

    print("2. Removing empty documents...")

    documents = remove_empty_documents(
        documents
    )

    print(
        f"{len(documents)} non-empty page(s) remain.\n"
    )

    # -----------------------------------------------------
    # 3. Split pages into smaller chunks
    # -----------------------------------------------------

    print("3. Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"Created {len(chunks)} chunk(s).\n"
    )

    # -----------------------------------------------------
    # 4. Convert LangChain Documents
    #    into our DocumentChunk model
    # -----------------------------------------------------

    print(
        "4. Converting chunks to DocumentChunk objects..."
    )

    document_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_id = chunk.metadata.get(
            "chunk_id",
            f"chunk-{index:05d}",
        )

        document_chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=chunk.page_content,
            metadata=chunk.metadata,
        )

        document_chunks.append(
            document_chunk
        )

    print(
        f"Created {len(document_chunks)} "
        "DocumentChunk object(s).\n"
    )

    # -----------------------------------------------------
    # 5. Initialize Chroma vector store
    # -----------------------------------------------------

    print("5. Initializing VectorStore...")

    settings = Settings()

    vector_store = VectorStore(
        settings
    )

    print(
        "VectorStore initialized successfully.\n"
    )

    # -----------------------------------------------------
    # 6. Store embeddings in ChromaDB
    # -----------------------------------------------------

    print(
        "6. Generating embeddings and storing chunks..."
    )

    vector_store.add_chunks(
        document_chunks
    )

    print(
        f"Stored chunks in ChromaDB: "
        f"{vector_store.count()}\n"
    )

    # -----------------------------------------------------
    # 7. Run semantic search
    # -----------------------------------------------------

    query = (
        "What is the main topic of the document?"
    )

    print("7. Running semantic search...\n")

    print(
        f"Query: {query}\n"
    )

    results = vector_store.search(
        query=query,
        top_k=3,
    )

    # -----------------------------------------------------
    # 8. Display search results
    # -----------------------------------------------------

    if not results:
        print(
            "No search results were returned."
        )

        return

    print(
        f"Top {len(results)} result(s):\n"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            "------------------------------------"
        )

        print(
            f"RESULT {index}"
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

        source_file = result.metadata.get(
            "source",
            "Unknown",
        )

        page_number = result.metadata.get(
            "page_number",
            "Unknown",
        )

        print(
            f"Source PDF: {source_file}"
        )

        print(
            f"Page: {page_number}"
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
        "PHASE 3 VECTOR SEARCH TEST COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
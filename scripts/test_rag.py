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
from app.rag import RAGService


def main():

    print("\n====================================")
    print("PHASE 6 - RAG TEST")
    print("====================================\n")

    # ---------------------------------------------
    # 1. Load settings
    # ---------------------------------------------

    settings = Settings()

    print("Using Ollama model:")
    print(settings.ollama_model)
    print()

    # ---------------------------------------------
    # 2. Load PDFs
    # ---------------------------------------------

    print("1. Loading PDF documents...")

    documents = load_pdfs(
        "data/raw"
    )

    print(
        f"Loaded {len(documents)} page(s)."
    )

    # ---------------------------------------------
    # 3. Remove empty documents
    # ---------------------------------------------

    documents = remove_empty_documents(
        documents
    )

    print(
        f"{len(documents)} non-empty page(s) remain.\n"
    )

    # ---------------------------------------------
    # 4. Chunk documents
    # ---------------------------------------------

    print("2. Creating document chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"Created {len(chunks)} chunk(s).\n"
    )

    # ---------------------------------------------
    # 5. Convert LangChain Documents
    #    into our DocumentChunk objects
    # ---------------------------------------------

    print(
        "3. Converting to DocumentChunk objects..."
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

    # ---------------------------------------------
    # 6. Initialize Vector Store
    # ---------------------------------------------

    print("4. Initializing vector store...")

    vector_store = VectorStore(
        settings
    )

    # ---------------------------------------------
    # 7. Initialize BM25 Store
    # ---------------------------------------------

    print("5. Initializing BM25 store...")

    bm25_store = BM25Store(
        settings
    )

    # ---------------------------------------------
    # 8. Index chunks
    # ---------------------------------------------

    print("6. Indexing document chunks...")

    vector_store.add_chunks(
        document_chunks
    )

    bm25_store.add_chunks(
        document_chunks
    )

    print(
        f"Vector chunks: {vector_store.count()}"
    )

    print(
        f"BM25 chunks: {len(bm25_store.chunks)}"
    )

    print()

    # ---------------------------------------------
    # 9. Create Hybrid Retriever
    # ---------------------------------------------

    print("7. Creating HybridRetriever...")

    retriever = HybridRetriever(
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # ---------------------------------------------
    # 10. Create RAG service
    # ---------------------------------------------

    print("8. Creating RAGService...")

    rag_service = RAGService(
        settings=settings,
        retriever=retriever,
    )

    print("RAG service ready.\n")

    # ---------------------------------------------
    # 11. Ask a question
    # ---------------------------------------------

    query = (
        "What are the requirements "
        "for remote work eligibility?"
    )

    print(
        "===================================="
    )

    print("QUESTION")

    print(
        "====================================\n"
    )

    print(query)

    print(
        "\nRetrieving documents and "
        "generating answer with Ollama...\n"
    )

    # ---------------------------------------------
    # 12. Full RAG execution
    # ---------------------------------------------

    result = rag_service.answer(
        query=query,
        top_k=5,
    )

    # ---------------------------------------------
    # 13. Display generated answer
    # ---------------------------------------------

    print(
        "===================================="
    )

    print("ANSWER")

    print(
        "====================================\n"
    )

    print(
        result.answer
    )

    # ---------------------------------------------
    # 14. Display retrieved sources
    # ---------------------------------------------

    print(
        "\n===================================="
    )

    print("SOURCES")

    print(
        "====================================\n"
    )

    if not result.sources:

        print(
            "No sources were returned."
        )

    else:

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
                f"Retrieval score: "
                f"{source.score:.4f}"
            )

            print(
                f"Retriever: "
                f"{source.source}"
            )

            print(
                f"Chunk ID: "
                f"{source.chunk_id}"
            )

            print()

    # ---------------------------------------------
    # 15. Finish
    # ---------------------------------------------

    print(
        "===================================="
    )

    print(
        "PHASE 6 RAG TEST COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
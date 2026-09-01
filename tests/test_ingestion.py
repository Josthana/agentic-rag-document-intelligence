from langchain_core.documents import Document

from app.ingestion import (
    remove_empty_documents,
    chunk_documents,
)


def test_remove_empty_documents():

    documents = [
        Document(
            page_content="Useful content",
            metadata={},
        ),
        Document(
            page_content="   ",
            metadata={},
        ),
        Document(
            page_content="More content",
            metadata={},
        ),
    ]

    cleaned = remove_empty_documents(
        documents
    )

    assert len(cleaned) == 2

    assert (
        cleaned[0].page_content
        == "Useful content"
    )


def test_chunk_documents():

    document = Document(
        page_content=(
            "Remote work is available to eligible "
            "employees. Employees must follow "
            "company security requirements. "
        )
        * 20,
        metadata={
            "source": "test.pdf",
            "page_number": 1,
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=200,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert "chunk_id" in chunk.metadata
        assert chunk.page_content
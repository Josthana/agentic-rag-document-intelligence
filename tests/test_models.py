from app.models import (
    DocumentChunk,
    RetrievedChunk,
    RAGResult,
)


def test_document_chunk():

    chunk = DocumentChunk(
        chunk_id="chunk-001",
        text="Remote work policy",
        metadata={
            "source": "test.pdf",
            "page_number": 1,
        },
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.text == "Remote work policy"
    assert chunk.metadata["source"] == "test.pdf"


def test_retrieved_chunk():

    chunk = RetrievedChunk(
        chunk_id="chunk-001",
        text="Remote work policy",
        metadata={
            "source": "test.pdf",
        },
        score=0.85,
        source="hybrid",
    )

    assert chunk.score == 0.85
    assert chunk.source == "hybrid"


def test_rag_result():

    result = RAGResult(
        query="What is remote work?",
        answer="Remote work is...",
        sources=[],
        used_query="remote work",
    )

    assert result.query == "What is remote work?"
    assert result.answer == "Remote work is..."
    assert result.used_query == "remote work"
    
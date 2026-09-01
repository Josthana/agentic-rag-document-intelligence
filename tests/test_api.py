from app.api import (
    app,
    root,
    QueryRequest,
    query_documents,
)

from app.models import (
    RAGResult,
    RetrievedChunk,
)


class FakeAgent:

    def answer(
        self,
        query,
    ):

        return RAGResult(
            query=query,
            answer=(
                "Remote work requires "
                "manager approval. [Source 1]"
            ),
            used_query=query,
            sources=[
                RetrievedChunk(
                    chunk_id="chunk-test",
                    text=(
                        "Remote work requires "
                        "manager approval."
                    ),
                    metadata={
                        "source": "remote_policy.pdf",
                        "page_number": 2,
                    },
                    score=0.9,
                    source="hybrid",
                )
            ],
        )


def test_app_title():

    assert (
        app.title
        == "Agentic RAG Document Intelligence API"
    )


def test_root_endpoint():

    response = root()

    assert (
        response["message"]
        == "Agentic RAG Document Intelligence API"
    )

    assert response["health"] == "/health"
    assert response["query"] == "/query"


def test_query_endpoint():

    app.state.agent = FakeAgent()

    request = QueryRequest(
        query="What is the remote work policy?"
    )

    response = query_documents(
        request
    )

    assert response.query == (
        "What is the remote work policy?"
    )

    assert "Remote work" in response.answer

    assert len(response.sources) == 1

    assert (
        response.sources[0].source
        == "remote_policy.pdf"
    )

    assert response.sources[0].page == 2

    assert (
        response.sources[0].retriever
        == "hybrid"
    )
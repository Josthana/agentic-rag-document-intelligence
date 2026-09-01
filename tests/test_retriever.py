from types import SimpleNamespace

from app.models import RetrievedChunk
from app.retriever import HybridRetriever


class FakeVectorStore:

    def search(
        self,
        query,
        top_k,
    ):

        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Remote work eligibility",
                metadata={
                    "source": "remote.pdf"
                },
                score=0.8,
                source="vector",
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                text="Employee performance",
                metadata={
                    "source": "handbook.pdf"
                },
                score=0.6,
                source="vector",
            ),
        ]


class FakeBM25Store:

    def search(
        self,
        query,
        top_k,
    ):

        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Remote work eligibility",
                metadata={
                    "source": "remote.pdf"
                },
                score=10.0,
                source="bm25",
            ),
            RetrievedChunk(
                chunk_id="chunk-3",
                text="Remote work rules",
                metadata={
                    "source": "remote.pdf"
                },
                score=5.0,
                source="bm25",
            ),
        ]


def test_hybrid_retriever():

    settings = SimpleNamespace(
        vector_top_k=5,
        bm25_top_k=5,
        final_top_k=5,
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    retriever = HybridRetriever(
        settings=settings,
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
    )

    results = retriever.retrieve(
        "remote work eligibility"
    )

    assert len(results) > 0

    # Chunk 1 appears in BOTH retrieval methods,
    # therefore it should rank first.
    assert results[0].chunk_id == "chunk-1"

    assert results[0].source == "hybrid"

    assert results[0].score > 0
from app.config import Settings
from app.models import RetrievedChunk
from app.vector_store import VectorStore
from app.bm25_store import BM25Store


class HybridRetriever:
    """
    Combine semantic vector retrieval and BM25
    keyword retrieval into one ranked result set.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks using both vector search
        and BM25, then combine their scores.
        """

        final_top_k = (
            top_k
            or self.settings.final_top_k
        )

        # -------------------------------------------------
        # 1. Semantic vector retrieval
        # -------------------------------------------------

        vector_results = (
            self.vector_store.search(
                query=query,
                top_k=self.settings.vector_top_k,
            )
        )

        # -------------------------------------------------
        # 2. Keyword BM25 retrieval
        # -------------------------------------------------

        bm25_results = (
            self.bm25_store.search(
                query=query,
                top_k=self.settings.bm25_top_k,
            )
        )

        # BM25 scores are not naturally between 0 and 1.
        # Normalize them before combining with vector scores.

        normalized_bm25 = (
            self._normalize_bm25(
                bm25_results
            )
        )

        # -------------------------------------------------
        # 3. Merge results using chunk_id
        # -------------------------------------------------

        combined: dict[str, dict] = {}

        for result in vector_results:

            combined[result.chunk_id] = {
                "chunk": result,
                "vector_score": self._clamp_score(
                    result.score
                ),
                "bm25_score": 0.0,
            }

        for result in normalized_bm25:

            if result.chunk_id not in combined:

                combined[result.chunk_id] = {
                    "chunk": result,
                    "vector_score": 0.0,
                    "bm25_score": result.score,
                }

            else:

                combined[
                    result.chunk_id
                ]["bm25_score"] = (
                    result.score
                )

        # -------------------------------------------------
        # 4. Weighted score fusion
        # -------------------------------------------------

        final_results = []

        for item in combined.values():

            vector_score = item[
                "vector_score"
            ]

            bm25_score = item[
                "bm25_score"
            ]

            final_score = (
                self.settings.vector_weight
                * vector_score
                +
                self.settings.bm25_weight
                * bm25_score
            )

            chunk = item["chunk"]

            final_results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=final_score,
                    source="hybrid",
                )
            )

        # -------------------------------------------------
        # 5. Rank highest score first
        # -------------------------------------------------

        final_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return final_results[
            :final_top_k
        ]

    @staticmethod
    def _normalize_bm25(
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Convert BM25 scores into the 0-1 range.
        """

        if not results:
            return []

        values = [
            result.score
            for result in results
        ]

        minimum = min(values)
        maximum = max(values)

        normalized = []

        for result in results:

            if maximum == minimum:

                score = (
                    1.0
                    if maximum > 0
                    else 0.0
                )

            else:

                score = (
                    result.score - minimum
                ) / (
                    maximum - minimum
                )

            normalized.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=result.metadata,
                    score=score,
                    source="bm25",
                )
            )

        return normalized

    @staticmethod
    def _clamp_score(
        score: float,
    ) -> float:
        """
        Keep vector scores between 0 and 1.
        """

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )
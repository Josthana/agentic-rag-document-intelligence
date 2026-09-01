import json
import re

from dataclasses import asdict

import numpy as np

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models import (
    DocumentChunk,
    RetrievedChunk,
)


class BM25Store:
    """
    Keyword-based retrieval store using BM25.
    """

    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

        self.chunks: list[DocumentChunk] = []

        self.bm25: BM25Okapi | None = None

        # Load an existing persisted BM25 corpus,
        # if one is available.
        self._load()

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:
        """
        Convert text into lowercase word tokens.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _build_index(self) -> None:
        """
        Build the BM25 index from the current chunks.
        """

        if not self.chunks:
            self.bm25 = None
            return

        tokenized_corpus = [
            self.tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Add or update chunks and rebuild the BM25 index.
        """

        existing = {
            chunk.chunk_id: chunk
            for chunk in self.chunks
        }

        for chunk in chunks:
            existing[chunk.chunk_id] = chunk

        self.chunks = list(
            existing.values()
        )

        self._build_index()

        self._save()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search the BM25 index using keyword relevance.
        """

        if self.bm25 is None:
            return []

        top_k = (
            top_k
            or self.settings.bm25_top_k
        )

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        if len(scores) == 0:
            return []

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                    source="bm25",
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _save(self) -> None:
        """
        Persist chunks to JSON so the BM25 index
        can be rebuilt on future runs.
        """

        path = self.settings.bm25_index_file

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = [
            asdict(chunk)
            for chunk in self.chunks
        ]

        path.write_text(
            json.dumps(
                records,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """
        Load persisted chunks and rebuild the BM25 index.
        """

        path = self.settings.bm25_index_file

        if not path.exists():
            return

        records = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.chunks = [
            DocumentChunk(**record)
            for record in records
        ]

        self._build_index()
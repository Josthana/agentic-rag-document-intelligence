from sentence_transformers import SentenceTransformer

import chromadb

from app.config import Settings
from app.models import DocumentChunk, RetrievedChunk


class VectorStore:

    def __init__(self, settings: Settings):

        self.settings = settings

        self.embedding_model = SentenceTransformer(
            settings.embedding_model
        )

        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=settings.chroma_collection
            )
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:

        if not chunks:
            return

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        metadata = [
            chunk.metadata
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:

        if self.collection.count() == 0:
            return []

        top_k = (
            top_k
            or self.settings.vector_top_k
        )

        query_embedding = (
            self.embedding_model.encode(
                query,
                normalize_embeddings=True,
            )
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results["ids"][0]

        documents = (
            results.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            results.get(
                "distances",
                [[]],
            )[0]
        )

        retrieved = []

        for (
            chunk_id,
            text,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            # Smaller vector distance = better.
            # Convert it into a simple larger-is-better score.
            score = 1.0 / (
                1.0 + float(distance)
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata or {},
                    score=score,
                    source="vector",
                )
            )

        return retrieved

    def count(self) -> int:
        return self.collection.count()
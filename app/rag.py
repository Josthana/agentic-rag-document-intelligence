import requests

from app.config import Settings
from app.models import (
    RAGResult,
    RetrievedChunk,
)
from app.retriever import HybridRetriever


class OllamaClient:

    def __init__(self, settings: Settings):

        self.settings = settings

    def chat(
        self,
        messages: list[dict],
    ) -> str:

        url = (
            f"{self.settings.ollama_base_url}"
            "/api/chat"
        )

        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.settings.ollama_timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "message"
        ]["content"].strip()


class RAGService:

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
    ):

        self.settings = settings

        self.retriever = retriever

        self.llm = OllamaClient(
            settings
        )

    def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RAGResult:

        sources = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return self.generate_from_context(
            query=query,
            sources=sources,
            used_query=query,
        )

    def generate_from_context(
        self,
        query: str,
        sources: list[RetrievedChunk],
        used_query: str | None = None,
    ) -> RAGResult:

        if not sources:

            return RAGResult(
                query=query,
                answer=(
                    "I could not find relevant "
                    "information in the indexed "
                    "documents."
                ),
                sources=[],
                used_query=used_query,
            )

        context = self._build_context(
            sources
        )

        system_prompt = """
You are a document intelligence assistant.

Answer the user's question using only the
document context provided.

Rules:
1. Do not invent information.
2. If the answer is not supported by the context,
   say that the documents do not contain enough
   information.
3. Cite document evidence using [Source 1],
   [Source 2], etc.
4. Be concise but complete.
5. Prefer facts from the supplied context over
   general knowledge.
""".strip()

        user_prompt = f"""
QUESTION:

{query}


DOCUMENT CONTEXT:

{context}


Answer the question using the document context.
""".strip()

        answer = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            used_query=used_query,
        )

    @staticmethod
    def _build_context(
        sources: list[RetrievedChunk],
    ) -> str:

        sections = []

        for index, source in enumerate(
            sources,
            start=1,
        ):

            filename = source.metadata.get(
                "filename",
                "unknown"
            )

            page = source.metadata.get(
                "page"
            )

            location = (
                f"File: {filename}"
            )

            if page:
                location += (
                    f", Page: {page}"
                )

            sections.append(
                f"""
[Source {index}]
{location}
Retrieval score: {source.score:.4f}

{source.text}
""".strip()
            )

        return "\n\n".join(
            sections
        )
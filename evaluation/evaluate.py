import json
import time

from datetime import datetime
from pathlib import Path
from statistics import mean

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
from app.graph import AgenticRAGService


QUESTIONS_FILE = Path(
    "evaluation/questions.json"
)

RESULTS_DIR = Path(
    "evaluation/results"
)


# =========================================================
# BUILD COMPLETE AGENTIC RAG SYSTEM
# =========================================================


def build_agent() -> AgenticRAGService:
    """
    Build the same Agentic RAG pipeline used
    by the application.
    """

    settings = Settings()

    print("\n====================================")
    print("BUILDING EVALUATION PIPELINE")
    print("====================================\n")

    # -----------------------------------------------------
    # 1. Load PDFs
    # -----------------------------------------------------

    print("[Evaluation] Loading PDFs...")

    documents = load_pdfs(
        "data/raw"
    )

    documents = remove_empty_documents(
        documents
    )

    print(
        f"[Evaluation] "
        f"{len(documents)} non-empty page(s)."
    )

    # -----------------------------------------------------
    # 2. Chunk documents
    # -----------------------------------------------------

    print(
        "[Evaluation] Creating chunks..."
    )

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=200,
    )

    print(
        f"[Evaluation] "
        f"{len(chunks)} chunk(s)."
    )

    # -----------------------------------------------------
    # 3. Convert into DocumentChunk objects
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 4. Build retrieval stores
    # -----------------------------------------------------

    print(
        "[Evaluation] Initializing retrieval stores..."
    )

    vector_store = VectorStore(
        settings
    )

    bm25_store = BM25Store(
        settings
    )

    vector_store.add_chunks(
        document_chunks
    )

    bm25_store.add_chunks(
        document_chunks
    )

    print(
        f"[Evaluation] Vector chunks: "
        f"{vector_store.count()}"
    )

    print(
        f"[Evaluation] BM25 chunks: "
        f"{len(bm25_store.chunks)}"
    )

    # -----------------------------------------------------
    # 5. Hybrid retriever
    # -----------------------------------------------------

    retriever = HybridRetriever(
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # -----------------------------------------------------
    # 6. LangGraph agent
    # -----------------------------------------------------

    agent = AgenticRAGService(
        settings=settings,
        retriever=retriever,
    )

    print(
        "\n[Evaluation] Agent ready.\n"
    )

    return agent


# =========================================================
# LOAD QUESTIONS
# =========================================================


def load_questions() -> list[dict]:
    """
    Load evaluation questions from JSON.
    """

    if not QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation file not found: "
            f"{QUESTIONS_FILE}"
        )

    return json.loads(
        QUESTIONS_FILE.read_text(
            encoding="utf-8"
        )
    )


# =========================================================
# SOURCE MATCH METRIC
# =========================================================


def calculate_source_hit(
    returned_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Return 1.0 when at least one expected
    document appears in retrieved sources.
    """

    if not expected_sources:
        return 1.0

    returned = {
        source.lower()
        for source in returned_sources
    }

    expected = {
        source.lower()
        for source in expected_sources
    }

    return (
        1.0
        if returned.intersection(expected)
        else 0.0
    )


# =========================================================
# KEYWORD COVERAGE METRIC
# =========================================================


def calculate_keyword_recall(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    Measure how many expected keywords
    appear in the generated answer.
    """

    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()

    matches = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return (
        matches
        / len(expected_keywords)
    )


# =========================================================
# CITATION METRIC
# =========================================================


def calculate_citation_score(
    answer: str,
) -> float:
    """
    Check whether the generated answer includes
    at least one [Source N] citation.
    """

    return (
        1.0
        if "[Source " in answer
        else 0.0
    )


# =========================================================
# OVERALL SCORE
# =========================================================


def calculate_overall_score(
    source_hit: float,
    keyword_recall: float,
    citation_score: float,
    has_keywords: bool,
) -> float:
    """
    Produce a simple 0-1 evaluation score.

    When expected keywords exist:
        40% source correctness
        40% keyword coverage
        20% citation presence

    Otherwise:
        70% source correctness
        30% citation presence
    """

    if has_keywords:

        return (
            0.40 * source_hit
            + 0.40 * keyword_recall
            + 0.20 * citation_score
        )

    return (
        0.70 * source_hit
        + 0.30 * citation_score
    )


# =========================================================
# EVALUATE ONE QUESTION
# =========================================================


def evaluate_question(
    agent: AgenticRAGService,
    item: dict,
) -> dict:
    """
    Run one evaluation question.
    """

    question = item["question"]

    print(
        "\n------------------------------------"
    )

    print(
        f"Evaluating: {item['id']}"
    )

    print(
        f"Question: {question}"
    )

    start_time = time.perf_counter()

    result = agent.answer(
        question
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    returned_sources = [
        source.metadata.get(
            "source",
            "Unknown",
        )
        for source in result.sources
    ]

    expected_sources = item.get(
        "expected_sources",
        [],
    )

    expected_keywords = item.get(
        "expected_keywords",
        [],
    )

    source_hit = calculate_source_hit(
        returned_sources,
        expected_sources,
    )

    keyword_recall = (
        calculate_keyword_recall(
            result.answer,
            expected_keywords,
        )
    )

    citation_score = (
        calculate_citation_score(
            result.answer
        )
    )

    overall_score = (
        calculate_overall_score(
            source_hit=source_hit,
            keyword_recall=keyword_recall,
            citation_score=citation_score,
            has_keywords=bool(
                expected_keywords
            ),
        )
    )

    retrieval_scores = [
        source.score
        for source in result.sources
    ]

    average_retrieval_score = (
        mean(retrieval_scores)
        if retrieval_scores
        else 0.0
    )

    print(
        f"Source hit: {source_hit:.2f}"
    )

    print(
        f"Keyword recall: "
        f"{keyword_recall:.2f}"
    )

    print(
        f"Citation score: "
        f"{citation_score:.2f}"
    )

    print(
        f"Overall score: "
        f"{overall_score:.2f}"
    )

    print(
        f"Latency: {elapsed:.2f}s"
    )

    return {
        "id": item["id"],
        "question": question,

        "answer": result.answer,

        "used_query": (
            result.used_query
        ),

        "expected_sources": (
            expected_sources
        ),

        "returned_sources": (
            returned_sources
        ),

        "expected_keywords": (
            expected_keywords
        ),

        "source_hit": (
            round(source_hit, 4)
        ),

        "keyword_recall": (
            round(keyword_recall, 4)
        ),

        "citation_score": (
            round(citation_score, 4)
        ),

        "average_retrieval_score": (
            round(
                average_retrieval_score,
                4,
            )
        ),

        "overall_score": (
            round(
                overall_score,
                4,
            )
        ),

        "latency_seconds": (
            round(
                elapsed,
                3,
            )
        ),

        "sources": [
            {
                "chunk_id": source.chunk_id,

                "source": (
                    source.metadata.get(
                        "source",
                        "Unknown",
                    )
                ),

                "page": (
                    source.metadata.get(
                        "page_number"
                    )
                ),

                "score": round(
                    source.score,
                    4,
                ),

                "retriever": (
                    source.source
                ),
            }

            for source in result.sources
        ],
    }


# =========================================================
# SUMMARY
# =========================================================


def build_summary(
    results: list[dict],
) -> dict:
    """
    Calculate aggregate evaluation metrics.
    """

    if not results:
        return {}

    return {
        "questions_evaluated": (
            len(results)
        ),

        "average_overall_score": round(
            mean(
                result[
                    "overall_score"
                ]
                for result in results
            ),
            4,
        ),

        "source_hit_rate": round(
            mean(
                result[
                    "source_hit"
                ]
                for result in results
            ),
            4,
        ),

        "average_keyword_recall": round(
            mean(
                result[
                    "keyword_recall"
                ]
                for result in results
            ),
            4,
        ),

        "citation_rate": round(
            mean(
                result[
                    "citation_score"
                ]
                for result in results
            ),
            4,
        ),

        "average_retrieval_score": round(
            mean(
                result[
                    "average_retrieval_score"
                ]
                for result in results
            ),
            4,
        ),

        "average_latency_seconds": round(
            mean(
                result[
                    "latency_seconds"
                ]
                for result in results
            ),
            3,
        ),
    }


# =========================================================
# SAVE RESULTS
# =========================================================


def save_results(
    results: list[dict],
    summary: dict,
) -> Path:
    """
    Save evaluation output to JSON.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        RESULTS_DIR
        / f"evaluation_{timestamp}.json"
    )

    payload = {
        "timestamp": timestamp,
        "summary": summary,
        "results": results,
    }

    output_file.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_file


# =========================================================
# MAIN
# =========================================================


def main():

    print(
        "\n===================================="
    )

    print(
        "PHASE 9 - RAG EVALUATION"
    )

    print(
        "===================================="
    )

    agent = build_agent()

    questions = load_questions()

    print(
        f"\nLoaded "
        f"{len(questions)} "
        "evaluation question(s)."
    )

    results = []

    for item in questions:

        result = evaluate_question(
            agent,
            item,
        )

        results.append(
            result
        )

    summary = build_summary(
        results
    )

    output_file = save_results(
        results,
        summary,
    )

    print(
        "\n===================================="
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "====================================\n"
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nResults saved to:"
        f"\n{output_file}"
    )

    print(
        "\n===================================="
    )

    print(
        "PHASE 9 EVALUATION COMPLETE"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()
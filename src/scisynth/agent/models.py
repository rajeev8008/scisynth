from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """Represents one citation linked to a retrieved chunk.

    Args:
        chunk_id: Retrieved chunk identifier.
        paper_id: Parsed paper identifier if present.
        paper_title: Document title when available from ingestion.
        snippet: Short snippet shown to the user.
        score: Retrieval score assigned to the chunk.
    Returns:
        None.
    """

    chunk_id: str
    paper_id: str
    paper_title: str | None
    snippet: str
    score: float


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Represents agent answer and evidence references.

    Args:
        question: Original user question.
        answer: Final natural language answer.
        citations: Evidence citations used in the answer.
        model: Model identifier used for generation.
        retrieval_hops_used: 1 after first retrieval only; 2 when a second hop ran.
    Returns:
        None.
    """

    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieval_hops_used: int = 1

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """Represents one citation linked to a retrieved chunk.

    Args:
        chunk_id: Retrieved chunk identifier.
        paper_id: Parsed paper identifier if present.
        snippet: Short snippet shown to the user.
        score: Retrieval score assigned to the chunk.
    Returns:
        None.
    """

    chunk_id: str
    paper_id: str
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
    Returns:
        None.
    """

    question: str
    answer: str
    citations: list[Citation]
    model: str

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Single retrieval result. Do not add fields without updating call sites."""

    id: str
    text: str
    score: float
    paper_id: str = ""
    paper_title: str | None = None


@runtime_checkable
class Retriever(Protocol):
    """Public retrieval API. Implementations: MockRetriever, LiveRetriever."""

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Return ranked chunks for the query."""
        ...

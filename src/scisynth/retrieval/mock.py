from __future__ import annotations

from scisynth.retrieval.contract import RetrievedChunk


class MockRetriever:
    """Deterministic retrieval for dev/unit tests. Not for benchmark metrics."""

    _FIXTURE_CHUNKS: tuple[RetrievedChunk, ...] = (
        RetrievedChunk(
            id="mock:chunk-1",
            text="Fixture passage A (mock retriever).",
            score=0.99,
            paper_id="mock-paper-a",
            paper_title="Mock Paper A",
        ),
        RetrievedChunk(
            id="mock:chunk-2",
            text="Fixture passage B (mock retriever).",
            score=0.95,
            paper_id="mock-paper-b",
            paper_title="Mock Paper B",
        ),
    )

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        _ = query
        return list(self._FIXTURE_CHUNKS[: max(0, top_k)])

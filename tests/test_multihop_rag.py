from __future__ import annotations

import scisynth.agent.service as service
from scisynth.agent.service import answer_question
from scisynth.config import Settings
from scisynth.retrieval.contract import RetrievedChunk


class _WeakThenStrongRetriever:
    """First call returns weak evidence; second returns stronger (simulates hop 2)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        if len(self.calls) == 1:
            return [
                RetrievedChunk(
                    id="p:chunk-0",
                    text="vague text",
                    score=0.05,
                    paper_id="p",
                    paper_title="P",
                ),
            ]
        return [
            RetrievedChunk(
                id="p:chunk-1",
                text="strong evidence for the user question",
                score=0.95,
                paper_id="p",
                paper_title="P",
            ),
        ]


def test_multihop_triggers_second_retrieval(monkeypatch) -> None:
    """When hop1 evidence is insufficient, a second retrieve runs and hops=2."""
    monkeypatch.setattr(
        service,
        "generate_answer_text",
        lambda *_a, **_k: "final",
    )
    r = _WeakThenStrongRetriever()
    settings = Settings(
        llm_api_key="x",
        llm_model="demo",
        rag_multi_hop=True,
        rag_max_hops=2,
        rag_evidence_min_chunks=2,
        rag_hop1_top_k=4,
        rag_hop2_top_k=8,
        answer_top_k=3,
    )
    result = answer_question("What is X?", settings=settings, retriever=r)
    assert result.retrieval_hops_used == 2
    assert len(r.calls) == 2
    assert "Supporting context from first retrieval" in r.calls[1][0]
    assert result.answer == "final"


def test_single_hop_when_evidence_strong(monkeypatch) -> None:
    """Strong first hop does not call retrieve twice."""
    monkeypatch.setattr(service, "generate_answer_text", lambda *_a, **_k: "ok")

    class _StrongRetriever:
        def __init__(self) -> None:
            self.n = 0

        def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
            self.n += 1
            return [
                RetrievedChunk(
                    id="a:chunk-0",
                    text="good",
                    score=0.95,
                    paper_id="a",
                    paper_title="A",
                ),
                RetrievedChunk(
                    id="a:chunk-1",
                    text="good2",
                    score=0.9,
                    paper_id="a",
                    paper_title="A",
                ),
            ]

    r = _StrongRetriever()
    settings = Settings(llm_api_key="x", llm_model="demo")
    result = answer_question("Q?", settings=settings, retriever=r)
    assert result.retrieval_hops_used == 1
    assert r.n == 1

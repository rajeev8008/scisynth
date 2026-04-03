from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient

from scisynth.agent.service import answer_question
from scisynth.config import Settings
from scisynth.eval.runner import run_frozen_eval
from scisynth.retrieval.contract import RetrievedChunk


class _FakeRetriever:
    """Simple fake retriever for deterministic agent tests."""

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        _ = query
        data = [
            RetrievedChunk(
                id="paper_a:chunk-0",
                text="Long documents need evidence across sections.",
                score=0.9,
                paper_id="paper_a",
                paper_title="Long Document QA",
            ),
            RetrievedChunk(
                id="paper_b:chunk-1",
                text="Better retrieval quality improves final synthesis.",
                score=0.8,
                paper_id="paper_b",
                paper_title="Retrieval Study",
            ),
        ]
        return data[:top_k]


def test_answer_question_returns_citations(monkeypatch) -> None:
    """Agent service returns answer text and citation metadata."""
    import scisynth.agent.service as service

    monkeypatch.setattr(
        service,
        "generate_answer_text",
        lambda *_args, **_kwargs: "Answer with [paper_a:chunk-0].",
    )
    settings = Settings(llm_api_key="x", llm_model="demo-model")
    result = answer_question(
        "What helps synthesis?",
        settings=settings,
        retriever=_FakeRetriever(),
    )
    assert "Answer with" in result.answer
    assert result.citations
    assert result.citations[0].paper_id == "paper_a"
    assert result.citations[0].paper_title == "Long Document QA"


def test_ask_endpoint_returns_answer_payload(monkeypatch) -> None:
    """POST /ask returns answer and citations with expected fields."""
    import scisynth.api.main as main

    class _StubResult:
        question = "What is retrieval?"
        answer = "Retrieved evidence answer."
        model = "stub-model"
        citations = []
        retrieval_hops_used = 1

    monkeypatch.setattr(main, "answer_question", lambda *_args, **_kwargs: _StubResult())
    with TestClient(main.app) as client:
        resp = client.post("/ask", json={"question": "What is retrieval?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Retrieved evidence answer."
    assert data["model"] == "stub-model"


def test_frozen_eval_writes_timestamped_csv(tmp_path: Path, monkeypatch) -> None:
    """Frozen eval writes a CSV row for each question."""
    import scisynth.eval.runner as runner

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "question": "Question one?"}),
                json.dumps({"id": "q2", "question": "Question two?"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _StubAnswer:
        answer = "stub answer"
        citations = []
        model = "stub-model"
        retrieval_hops_used = 1

    monkeypatch.setattr(runner, "answer_question", lambda *_args, **_kwargs: _StubAnswer())
    settings = Settings(
        eval_questions_path=str(questions),
        eval_results_dir=str(tmp_path / "results"),
    )
    summary = run_frozen_eval(settings)
    out = Path(summary.output_csv)
    assert out.exists()
    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "keyword_overlap" in rows[0]
    assert "retrieval_hops_used" in rows[0]
    assert "rubric_keyword_coverage" in rows[0]


def test_ask_rejects_discovery_and_single_arxiv_together() -> None:
    """POST /ask cannot combine arxiv_discovery with arxiv_url_or_id."""
    from scisynth.api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/ask",
            json={
                "question": "What is attention?",
                "arxiv_discovery": True,
                "arxiv_url_or_id": "1706.03762",
            },
        )
    assert resp.status_code == 422


def test_ask_stream_index_mode_returns_sse(monkeypatch) -> None:
    """POST /ask/stream yields SSE lines when index retrieval + LLM stream are stubbed."""
    import scisynth.api.main as main
    from scisynth.retrieval.contract import RetrievedChunk

    ch = RetrievedChunk(
        id="paper:chunk-0",
        text="context text",
        score=0.9,
        paper_id="paper",
        paper_title="Title",
    )

    monkeypatch.setattr(
        main,
        "retrieve_chunks_for_answer",
        lambda *a, **k: ([ch], 1),
    )
    monkeypatch.setattr(
        main,
        "generate_answer_text_stream",
        lambda *a, **k: iter(["partial", " answer"]),
    )

    with TestClient(main.app) as client:
        resp = client.post("/ask/stream", json={"question": "Test question?"})
    assert resp.status_code == 200
    body = resp.text
    assert "meta" in body
    assert "token" in body
    assert "done" in body
    assert "partial" in body

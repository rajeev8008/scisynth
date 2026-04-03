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
            ),
            RetrievedChunk(
                id="paper_b:chunk-1",
                text="Better retrieval quality improves final synthesis.",
                score=0.8,
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


def test_ask_endpoint_returns_answer_payload(monkeypatch) -> None:
    """POST /ask returns answer and citations with expected fields."""
    import scisynth.api.main as main

    class _StubResult:
        question = "What is retrieval?"
        answer = "Retrieved evidence answer."
        model = "stub-model"
        citations = []

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

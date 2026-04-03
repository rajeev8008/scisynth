from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scisynth.agent import answer_question
from scisynth.config import Settings


@dataclass(frozen=True, slots=True)
class EvalRunSummary:
    """Summary describing one frozen evaluation run.

    Args:
        output_csv: Path to written CSV results.
        question_count: Number of processed questions.
    Returns:
        None.
    """

    output_csv: str
    question_count: int


def _keyword_overlap(question: str, answer: str) -> float:
    """Cheap token overlap score between question and answer (0..1)."""
    q = {w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 2}
    a = {w for w in re.findall(r"[A-Za-z0-9]+", answer.lower()) if len(w) > 2}
    if not q or not a:
        return 0.0
    inter = len(q & a)
    return inter / max(len(q), 1)


def _has_chunk_citation_markers(answer: str) -> bool:
    """True if answer looks like it cites chunk ids (rough rubric signal)."""
    return bool(re.search(r"\[[^\]\n]*:chunk-[^\]\n]*\]", answer))


def _rubric_keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of optional rubric keywords present in answer (case-insensitive)."""
    if not keywords:
        return 1.0
    low = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in low)
    return round(hits / len(keywords), 4)


def run_frozen_eval(settings: Settings) -> EvalRunSummary:
    """Run question-answering over a frozen question set and log CSV output.

    Args:
        settings: Runtime settings with eval paths and model config.
    Returns:
        Summary with result path and processed count.
    """
    questions = _read_questions(Path(settings.eval_questions_path))
    rows: list[dict[str, object]] = []
    for item in questions:
        result = answer_question(item["question"], settings=settings)
        overlap = _keyword_overlap(item["question"], result.answer)
        rubric_keys = item.get("rubric_keywords") or []
        rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "citation_count": len(result.citations),
                "retrieval_hops_used": result.retrieval_hops_used,
                "keyword_overlap": round(overlap, 4),
                "has_chunk_citation_markers": _has_chunk_citation_markers(result.answer),
                "rubric_keyword_coverage": _rubric_keyword_coverage(
                    result.answer,
                    [str(x) for x in rubric_keys] if isinstance(rubric_keys, list) else [],
                ),
                "model": result.model,
            }
        )
    out_path = _write_results(Path(settings.eval_results_dir), rows)
    return EvalRunSummary(output_csv=str(out_path), question_count=len(rows))


def _read_questions(path: Path) -> list[dict[str, object]]:
    """Read JSONL frozen eval questions.

    Each line: ``id``, ``question``, optional ``rubric_keywords`` (list of strings).
    """
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row["id"])
        q = str(row["question"])
        out: dict[str, object] = {"id": rid, "question": q}
        if "rubric_keywords" in row and isinstance(row["rubric_keywords"], list):
            out["rubric_keywords"] = row["rubric_keywords"]
        rows.append(out)
    return rows


def _write_results(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    """Write eval outputs to a timestamped CSV file.

    Args:
        output_dir: Destination directory for eval result files.
        rows: Flat row dictionaries for CSV output.
    Returns:
        Full path to generated CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"phase4_eval_{stamp}.csv"
    fieldnames = [
        "question_id",
        "question",
        "answer",
        "citation_count",
        "retrieval_hops_used",
        "keyword_overlap",
        "has_chunk_citation_markers",
        "rubric_keyword_coverage",
        "model",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

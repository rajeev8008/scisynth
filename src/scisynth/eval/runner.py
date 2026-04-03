from __future__ import annotations

import csv
import json
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
        rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "citation_count": len(result.citations),
                "model": result.model,
            }
        )
    out_path = _write_results(Path(settings.eval_results_dir), rows)
    return EvalRunSummary(output_csv=str(out_path), question_count=len(rows))


def _read_questions(path: Path) -> list[dict[str, str]]:
    """Read JSONL frozen eval questions.

    Args:
        path: JSONL file path containing id and question keys.
    Returns:
        List of question dictionaries.
    """
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"id": str(row["id"]), "question": str(row["question"])})
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
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["question_id", "question", "answer", "citation_count", "model"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path

from __future__ import annotations

import json
from pathlib import Path

from scisynth.ingestion.schema import PaperDocument


def write_papers_to_raw_dir(
    root: Path,
    dataset_id: str,
    documents: list[PaperDocument],
) -> Path:
    """Write paper bodies and metadata.jsonl under root/dataset_id.

    Args:
        root: Base directory (e.g. data/raw/arxiv).
        dataset_id: Subfolder name for this ingest run.
        documents: Loaded papers to persist.
    Returns:
        Directory that was written.
    """
    out = (root / dataset_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for doc in documents:
        stem = _safe_filename_stem(doc.paper_id)
        md_path = out / f"{stem}.md"
        md_path.write_text(doc.text, encoding="utf-8")
        rows.append(
            {
                "paper_id": doc.paper_id,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "topic": doc.topic,
                "abstract": doc.abstract,
            }
        )
    meta_path = out / "metadata.jsonl"
    meta_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return out


def _safe_filename_stem(paper_id: str) -> str:
    """Make a filesystem-safe stem from a paper id.

    Args:
        paper_id: Raw identifier.
    Returns:
        Safe filename stem.
    """
    cleaned = paper_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    return cleaned or "paper"

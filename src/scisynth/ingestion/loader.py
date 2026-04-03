from __future__ import annotations

import json
from pathlib import Path

from scisynth.ingestion.schema import PaperDocument

_METADATA_FILE = "metadata.jsonl"
_SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(dataset_path: Path) -> list[PaperDocument]:
    """Load dataset files and metadata into paper documents.

    Args:
        dataset_path: Directory containing source files and metadata.jsonl.
    Returns:
        List of loaded paper documents sorted by paper_id.
    """
    metadata_by_id = _read_metadata(dataset_path / _METADATA_FILE)
    docs: list[PaperDocument] = []
    for source_file in sorted(dataset_path.iterdir()):
        if source_file.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        paper_id = source_file.stem
        meta = metadata_by_id.get(paper_id, _default_metadata(paper_id))
        text = source_file.read_text(encoding="utf-8")
        docs.append(
            PaperDocument(
                paper_id=paper_id,
                title=meta["title"],
                authors=meta["authors"],
                year=int(meta["year"]),
                topic=meta["topic"],
                abstract=meta["abstract"],
                source_path=str(source_file),
                text=text,
            )
        )
    return sorted(docs, key=lambda item: item.paper_id)


def _read_metadata(metadata_path: Path) -> dict[str, dict[str, str | int]]:
    """Read metadata.jsonl into a map by paper id.

    Args:
        metadata_path: Path to metadata.jsonl file.
    Returns:
        Metadata map keyed by paper_id, or empty when missing.
    """
    if not metadata_path.exists():
        return {}
    by_id: dict[str, dict[str, str | int]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[str(row["paper_id"])] = row
    return by_id


def _default_metadata(paper_id: str) -> dict[str, str | int]:
    """Build fallback metadata when none is provided.

    Args:
        paper_id: Paper identifier.
    Returns:
        Metadata dictionary with required keys.
    """
    return {
        "paper_id": paper_id,
        "title": paper_id,
        "authors": "unknown",
        "year": 0,
        "topic": "unspecified",
        "abstract": "",
    }

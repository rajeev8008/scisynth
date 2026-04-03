from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scisynth.ingestion.schema import ChunkRecord, PaperDocument


def write_ingestion_outputs(
    output_dir: Path,
    documents: list[PaperDocument],
    chunks: list[ChunkRecord],
    *,
    dataset_id: str,
    dataset_profile: str,
    source_path: str,
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    """Write ingestion artifacts to a versioned output directory.

    Args:
        output_dir: Parent output directory from config.
        documents: Loaded source documents.
        chunks: Produced chunk records.
        dataset_id: Dataset id/version label.
        dataset_profile: Active profile name.
        source_path: Source dataset path.
        manifest_extra: Optional extra manifest fields (source-specific).
    Returns:
        Final dataset output directory path.
    """
    dataset_output = output_dir / dataset_id
    dataset_output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dataset_output / "documents.jsonl", [asdict(item) for item in documents])
    _write_jsonl(dataset_output / "chunks.jsonl", [asdict(item) for item in chunks])
    manifest: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_profile": dataset_profile,
        "source_path": source_path,
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (dataset_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return dataset_output


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a JSONL file from row dictionaries.

    Args:
        path: Output file path.
        rows: Row dictionaries to serialize.
    Returns:
        None.
    """
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")

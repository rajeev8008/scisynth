from __future__ import annotations

import json
from pathlib import Path

from scisynth.ingestion.schema import ChunkRecord


def load_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    """Load chunk records written by ingestion (chunks.jsonl)."""
    if not path.is_file():
        return []
    rows: list[ChunkRecord] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append(
            ChunkRecord(
                chunk_id=str(d["chunk_id"]),
                paper_id=str(d["paper_id"]),
                chunk_index=int(d["chunk_index"]),
                text=str(d["text"]),
            )
        )
    return rows

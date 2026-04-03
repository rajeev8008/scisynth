from __future__ import annotations

import json
from pathlib import Path


def load_paper_meta(path: Path) -> dict[str, str]:
    """Load paper_id -> title from documents.jsonl (ingestion output)."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pid = str(d.get("paper_id", ""))
        if pid:
            out[pid] = str(d.get("title", "")).strip() or pid
    return out

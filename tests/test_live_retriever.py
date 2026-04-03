from __future__ import annotations

import json
from pathlib import Path

from scisynth.config import Settings
from scisynth.retrieval.live import LiveRetriever


def _write_chunks(path: Path, rows: list[dict[str, object]]) -> None:
    lines = "\n".join(json.dumps(r, ensure_ascii=True) for r in rows)
    path.write_text(f"{lines}\n" if lines else "", encoding="utf-8")


def test_live_retriever_bm25_ranks_by_query(tmp_path: Path) -> None:
    """Live retriever returns chunks scored by BM25 relevance."""
    ds = tmp_path / "my-dataset"
    ds.mkdir(parents=True)
    _write_chunks(
        ds / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "chunk_index": 0,
                "text": "The cat sat on the mat.",
            },
            {
                "chunk_id": "c2",
                "paper_id": "p1",
                "chunk_index": 1,
                "text": "Neural networks learn representations from data.",
            },
            {
                "chunk_id": "c3",
                "paper_id": "p2",
                "chunk_index": 0,
                "text": "Cooking recipes for pasta and sauce.",
            },
        ],
    )
    settings = Settings(
        retriever_mode="live",
        ingestion_output_path=str(tmp_path),
        dataset_id="my-dataset",
    )
    r = LiveRetriever(settings=settings)
    out = r.retrieve("neural network learn", top_k=2)
    assert len(out) == 2
    assert out[0].id == "c2"
    assert out[0].paper_id == "p1"
    assert out[0].score >= out[1].score


def test_live_retriever_loads_paper_titles_from_documents(tmp_path: Path) -> None:
    """documents.jsonl supplies titles for live retrieval results."""
    ds = tmp_path / "ds-titles"
    ds.mkdir(parents=True)
    _write_chunks(
        ds / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "chunk_index": 0,
                "text": "Unique neuroplasticity evidence here.",
            },
        ],
    )
    doc_lines = [
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Neuroscience Review",
                "authors": "X",
                "year": 2020,
                "topic": "ns",
                "abstract": "",
                "source_path": "x",
                "text": "",
            },
            ensure_ascii=True,
        )
    ]
    (ds / "documents.jsonl").write_text("\n".join(doc_lines) + "\n", encoding="utf-8")
    settings = Settings(
        retriever_mode="live",
        ingestion_output_path=str(tmp_path),
        dataset_id="ds-titles",
    )
    r = LiveRetriever(settings=settings)
    out = r.retrieve("neuroplasticity", top_k=1)
    assert len(out) == 1
    assert out[0].paper_title == "Neuroscience Review"


def test_live_retriever_missing_chunks_returns_empty(tmp_path: Path) -> None:
    """When chunks.jsonl is absent, retrieval returns an empty list."""
    settings = Settings(
        retriever_mode="live",
        ingestion_output_path=str(tmp_path),
        dataset_id="missing",
    )
    r = LiveRetriever(settings=settings)
    assert r.retrieve("anything") == []

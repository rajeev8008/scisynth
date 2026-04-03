from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch

from scisynth.config import Settings
from scisynth.ingestion.arxiv_loader import load_arxiv_documents
from scisynth.ingestion.hf_loader import _qasper_row_to_doc
from scisynth.ingestion.loader import load_documents
from scisynth.ingestion.pipeline import run_ingestion
from scisynth.ingestion.raw_snapshot import write_papers_to_raw_dir
from scisynth.ingestion.schema import PaperDocument
from scisynth.ingestion.transform import chunk_documents


def test_fixture_loads_required_metadata_keys() -> None:
    """Fixture dataset loads with required metadata fields.

    Args:
        None.
    Returns:
        None.
    """
    fixture_dir = Path(__file__).parent.parent / "src" / "scisynth" / "ingestion" / "fixtures" / "v1"
    docs = load_documents(fixture_dir)
    assert docs
    first = docs[0]
    assert first.paper_id
    assert first.title
    assert first.authors
    assert isinstance(first.year, int)
    assert first.topic
    assert isinstance(first.abstract, str)


def test_ingestion_writes_versioned_output(tmp_path: Path) -> None:
    """Ingestion pipeline writes chunks and manifest files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    """
    fixture_dir = Path(__file__).parent.parent / "src" / "scisynth" / "ingestion" / "fixtures" / "v1"
    settings = Settings(
        dataset_profile="fixture",
        dataset_id="fixture-v1-test",
        dataset_fixture_path=str(fixture_dir),
        ingestion_output_path=str(tmp_path),
        chunk_size=120,
        chunk_overlap=20,
    )
    stats = run_ingestion(settings)
    output_dir = Path(stats.output_path)
    assert output_dir.name == "fixture-v1-test"
    assert (output_dir / "documents.jsonl").exists()
    assert (output_dir / "chunks.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_source"] == "local"
    assert manifest["chunk_size"] == 120
    assert manifest["chunk_overlap"] == 20


def test_chunking_returns_non_empty_records() -> None:
    """Chunking produces at least one chunk for fixture docs.

    Args:
        None.
    Returns:
        None.
    """
    fixture_dir = Path(__file__).parent.parent / "src" / "scisynth" / "ingestion" / "fixtures" / "v1"
    docs = load_documents(fixture_dir)
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(chunk.text for chunk in chunks)


def test_arxiv_loader_maps_results_without_network(monkeypatch: MonkeyPatch) -> None:
    """arXiv loader maps API results into PaperDocument records.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    Returns:
        None.
    """
    import scisynth.ingestion.arxiv_loader as arxiv_loader

    class FakeClient:
        """Fake arXiv client yielding one deterministic result."""

        def results(self, _search):
            yield SimpleNamespace(
                title="Demo arXiv Paper",
                summary="Demo abstract text.",
                authors=[SimpleNamespace(name="Author One")],
                entry_id="https://arxiv.org/abs/1234.5678",
                published=None,
                get_short_id=lambda: "1234.5678",
            )

    monkeypatch.setattr(arxiv_loader.arxiv, "Client", lambda: FakeClient())
    settings = Settings(dataset_source="arxiv", arxiv_query="cat:cs.CL", arxiv_max_results=1)
    docs = load_arxiv_documents(settings)
    assert len(docs) == 1
    assert docs[0].paper_id == "1234.5678"
    assert docs[0].title == "Demo arXiv Paper"


def test_qasper_row_mapping_flattens_sections() -> None:
    """QASPER-style rows flatten section paragraphs into text.

    Args:
        None.
    Returns:
        None.
    """
    row = {
        "id": "paper-1",
        "title": "Title",
        "abstract": "Abstract.",
        "full_text": {
            "section_name": ["Intro"],
            "paragraphs": [["First paragraph.", "Second."]],
        },
    }
    doc = _qasper_row_to_doc(row)
    assert doc.paper_id == "paper-1"
    assert "Intro" in doc.text
    assert "First paragraph." in doc.text


def test_raw_snapshot_writes_markdown_bundle(tmp_path: Path) -> None:
    """Raw snapshot writes markdown files plus metadata.jsonl.

    Args:
        tmp_path: Temporary directory.
    Returns:
        None.
    """
    docs = [
        PaperDocument(
            paper_id="p/1",
            title="T",
            authors="A",
            year=2020,
            topic="t",
            abstract="ab",
            source_path="s",
            text="body",
        )
    ]
    out = write_papers_to_raw_dir(tmp_path, "run1", docs)
    assert (out / "p_1.md").exists()
    assert (out / "metadata.jsonl").read_text(encoding="utf-8").strip()


def test_huggingface_pipeline_branch_without_network(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Ingestion uses HF loader path and records provenance in manifest.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory.
    Returns:
        None.
    """
    fake_doc = PaperDocument(
        paper_id="hf-1",
        title="HF Title",
        authors="unknown",
        year=0,
        topic="qasper",
        abstract="Short abstract.",
        source_path="hf:test",
        text="paragraph " * 40,
    )

    def _fake_hf_loader(_settings: Settings) -> list[PaperDocument]:
        return [fake_doc]

    monkeypatch.setattr(
        "scisynth.ingestion.pipeline.load_hf_documents",
        _fake_hf_loader,
    )
    settings = Settings(
        dataset_source="huggingface",
        hf_preset="qasper",
        dataset_id="hf-ingest-test",
        ingestion_output_path=str(tmp_path),
        chunk_size=50,
        chunk_overlap=10,
    )
    stats = run_ingestion(settings)
    output_dir = Path(stats.output_path)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_source"] == "huggingface"
    assert manifest["hf_preset"] == "qasper"
    assert stats.document_count == 1
    assert stats.chunk_count >= 1

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scisynth.config import Settings
from scisynth.ingestion.arxiv_loader import load_arxiv_documents
from scisynth.ingestion.hf_loader import load_hf_documents
from scisynth.ingestion.loader import load_documents
from scisynth.ingestion.raw_snapshot import write_papers_to_raw_dir
from scisynth.ingestion.schema import PaperDocument
from scisynth.ingestion.transform import chunk_documents
from scisynth.ingestion.writer import write_ingestion_outputs


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """Summary returned by the ingestion pipeline.

    Args:
        dataset_id: Dataset version identifier.
        source_path: Source dataset directory.
        output_path: Final output directory.
        document_count: Number of loaded documents.
        chunk_count: Number of produced chunks.
    Returns:
        None.
    """

    dataset_id: str
    source_path: str
    output_path: str
    document_count: int
    chunk_count: int


def run_ingestion(settings: Settings) -> IngestionStats:
    """Run load, clean/chunk, and write pipeline for configured dataset.

    Args:
        settings: Application settings with dataset and chunk config.
    Returns:
        Ingestion summary with source and output counts/paths.
    """
    source_label, documents = _resolve_documents(settings)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    manifest_extra = _build_manifest_extra(settings)
    output_path = write_ingestion_outputs(
        Path(settings.ingestion_output_path),
        documents,
        chunks,
        dataset_id=settings.dataset_id,
        dataset_profile=settings.dataset_profile,
        source_path=source_label,
        manifest_extra=manifest_extra,
    )
    return IngestionStats(
        dataset_id=settings.dataset_id,
        source_path=source_label,
        output_path=str(output_path),
        document_count=len(documents),
        chunk_count=len(chunks),
    )


def _resolve_documents(settings: Settings) -> tuple[str, list[PaperDocument]]:
    """Resolve and load documents from selected ingestion source.

    Args:
        settings: Application settings.
    Returns:
        Source label and loaded paper documents.
    """
    if settings.dataset_source == "arxiv":
        source = f"arxiv:{settings.arxiv_query}"
        documents = load_arxiv_documents(settings)
        if settings.arxiv_persist_raw:
            raw_root = Path(settings.ingestion_raw_path) / "arxiv"
            write_papers_to_raw_dir(raw_root, settings.dataset_id, documents)
        return source, documents
    if settings.dataset_source == "huggingface":
        source = f"huggingface:{settings.hf_preset}"
        return source, load_hf_documents(settings)
    selected = (
        settings.dataset_fixture_path
        if settings.dataset_profile == "fixture"
        else settings.dataset_full_path
    )
    source_path = Path(selected).resolve()
    return str(source_path), load_documents(source_path)


def _build_manifest_extra(settings: Settings) -> dict[str, object]:
    """Assemble optional manifest fields for reproducibility.

    Args:
        settings: Application settings.
    Returns:
        Extra manifest key/value pairs.
    """
    extra: dict[str, object] = {
        "dataset_source": settings.dataset_source,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "ingestion_raw_path": settings.ingestion_raw_path,
    }
    if settings.dataset_source == "arxiv":
        extra["arxiv_query"] = settings.arxiv_query
        extra["arxiv_max_results"] = settings.arxiv_max_results
        extra["arxiv_persist_raw"] = settings.arxiv_persist_raw
    if settings.dataset_source == "huggingface":
        extra["hf_preset"] = settings.hf_preset
        extra["hf_split"] = settings.hf_split
        extra["hf_max_rows"] = settings.hf_max_rows
        extra["hf_qasper_revision"] = settings.hf_qasper_revision
        extra["hf_scifact_parquet_glob"] = settings.hf_scifact_parquet_glob
    return extra

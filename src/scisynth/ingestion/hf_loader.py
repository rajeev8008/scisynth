from __future__ import annotations

from typing import Any, Literal, cast

from scisynth.config import Settings
from scisynth.ingestion.schema import PaperDocument

HFPreset = Literal["qasper", "scifact_corpus"]


def load_hf_documents(settings: Settings) -> list[PaperDocument]:
    """Load a small HF eval slice and map rows to paper documents.

    Args:
        settings: Application settings (preset, split, caps, revisions).
    Returns:
        Paper documents sorted by paper_id.

    Raises:
        ImportError: When the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "Install datasets: pip install 'scisynth[hf]' or pip install datasets>=3.0"
        raise ImportError(msg) from exc

    preset = cast(HFPreset, settings.hf_preset)
    if preset == "qasper":
        docs = _load_qasper_rows(load_dataset, settings)
    else:
        docs = _load_scifact_corpus_rows(load_dataset, settings)
    docs.sort(key=lambda item: item.paper_id)
    return docs


def _load_qasper_rows(load_dataset: Any, settings: Settings) -> list[PaperDocument]:
    """Load QASPER rows from the parquet Hub revision.

    Args:
        load_dataset: ``datasets.load_dataset`` callable.
        settings: Application settings.
    Returns:
        Mapped paper documents.
    """
    split = _split_with_cap(settings.hf_split, settings.hf_max_rows)
    ds = load_dataset(
        "allenai/qasper",
        split=split,
        revision=settings.hf_qasper_revision,
    )
    return [_qasper_row_to_doc(cast(dict[str, Any], row)) for row in ds]


def _load_scifact_corpus_rows(load_dataset: Any, settings: Settings) -> list[PaperDocument]:
    """Load SciFact corpus shards from parquet on the Hub.

    Args:
        load_dataset: ``datasets.load_dataset`` callable.
        settings: Application settings.
    Returns:
        Mapped paper documents.
    """
    cap = max(0, settings.hf_max_rows)
    split_arg = f"train[:{cap}]" if cap else "train"
    ds = load_dataset(
        "parquet",
        data_files=settings.hf_scifact_parquet_glob,
        split=split_arg,
    )
    return [_scifact_corpus_row_to_doc(cast(dict[str, Any], row)) for row in ds]


def _split_with_cap(split: str, max_rows: int) -> str:
    """Build a datasets split string with an optional row cap.

    Args:
        split: Base split name (e.g. train).
        max_rows: Maximum rows (0 means no cap).
    Returns:
        Split string understood by Hugging Face Datasets.
    """
    if max_rows <= 0:
        return split
    return f"{split}[:{max_rows}]"


def _qasper_row_to_doc(row: dict[str, Any]) -> PaperDocument:
    """Map one QASPER example to PaperDocument.

    Args:
        row: Raw dataset row.
    Returns:
        Normalized paper document.
    """
    paper_id = str(row.get("id", "unknown"))
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    body = _flatten_qasper_full_text(row.get("full_text"))
    text = "\n\n".join(part for part in (title, abstract, body) if part.strip()).strip()
    authors = str(row.get("authors", "unknown") or "unknown")
    year = int(row.get("year", 0) or 0)
    return PaperDocument(
        paper_id=paper_id,
        title=title or paper_id,
        authors=authors,
        year=year,
        topic="qasper",
        abstract=abstract,
        source_path=f"hf:allenai/qasper:{paper_id}",
        text=text or abstract or title,
    )


def _flatten_qasper_full_text(full_text: Any) -> str:
    """Flatten QASPER full_text (sections + paragraphs) to plain text.

    Args:
        full_text: Dataset field (dict of lists or None).
    Returns:
        Joined body text.
    """
    if not isinstance(full_text, dict):
        return ""
    names = full_text.get("section_name") or []
    paras = full_text.get("paragraphs") or []
    if not isinstance(names, list) or not isinstance(paras, list):
        return ""
    parts: list[str] = []
    for name, section_paras in zip(names, paras):
        header = str(name).strip()
        if isinstance(section_paras, list):
            body = "\n".join(str(p) for p in section_paras if p)
        else:
            body = str(section_paras)
        chunk = f"{header}\n{body}".strip() if header else body
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


def _scifact_corpus_row_to_doc(row: dict[str, Any]) -> PaperDocument:
    """Map one SciFact corpus row to PaperDocument.

    Args:
        row: Raw corpus row.
    Returns:
        Normalized paper document.
    """
    doc_id = row.get("doc_id")
    paper_id = str(int(doc_id)) if doc_id is not None else "unknown"
    title = str(row.get("title", ""))
    abstract_field = row.get("abstract")
    if isinstance(abstract_field, list):
        abstract = " ".join(str(p) for p in abstract_field if p)
    else:
        abstract = str(abstract_field or "")
    text = "\n\n".join(part for part in (title, abstract) if part.strip()).strip()
    return PaperDocument(
        paper_id=f"scifact-{paper_id}",
        title=title or paper_id,
        authors="unknown",
        year=0,
        topic="scifact_corpus",
        abstract=abstract,
        source_path=f"hf:allenai/scifact:corpus:{paper_id}",
        text=text or abstract or title,
    )

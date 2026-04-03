from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaperDocument:
    """Represents one source paper loaded from the dataset.

    Args:
        paper_id: Stable paper identifier.
        title: Paper title.
        authors: Author list as a display string.
        year: Publication year.
        topic: Topic/category label.
        abstract: Short abstract text.
        source_path: Path to the source file.
        text: Full loaded paper text.
    Returns:
        None.
    """

    paper_id: str
    title: str
    authors: str
    year: int
    topic: str
    abstract: str
    source_path: str
    text: str


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Represents one cleaned chunk produced by ingestion.

    Args:
        chunk_id: Stable chunk identifier.
        paper_id: Parent paper identifier.
        chunk_index: Position within the paper.
        text: Chunk text content.
    Returns:
        None.
    """

    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str

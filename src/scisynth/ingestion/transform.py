from __future__ import annotations

import re

from scisynth.ingestion.schema import ChunkRecord, PaperDocument


def clean_text(text: str) -> str:
    """Normalize whitespace in source text.

    Args:
        text: Raw source text.
    Returns:
        Cleaned text with normalized spacing.
    """
    compact = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def chunk_documents(
    documents: list[PaperDocument], chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    """Split cleaned documents into fixed-size overlapping chunks.

    Args:
        documents: Loaded paper documents.
        chunk_size: Max chunk length in characters.
        chunk_overlap: Overlap length in characters.
    Returns:
        Flat chunk list for all documents.
    """
    chunks: list[ChunkRecord] = []
    for doc in documents:
        cleaned = clean_text(doc.text)
        chunks.extend(_chunk_single_document(doc.paper_id, cleaned, chunk_size, chunk_overlap))
    return chunks


def _chunk_single_document(
    paper_id: str, cleaned_text: str, chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    """Chunk one document into overlapping windows.

    Args:
        paper_id: Parent paper id.
        cleaned_text: Cleaned document text.
        chunk_size: Max chunk length in characters.
        chunk_overlap: Overlap length in characters.
    Returns:
        Ordered chunk list for the paper.
    """
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid chunk settings: require chunk_size > chunk_overlap >= 0.")
    step = chunk_size - chunk_overlap
    records: list[ChunkRecord] = []
    for idx, start in enumerate(range(0, len(cleaned_text), step)):
        stop = start + chunk_size
        text = cleaned_text[start:stop].strip()
        if text:
            records.append(
                ChunkRecord(
                    chunk_id=f"{paper_id}:chunk-{idx}",
                    paper_id=paper_id,
                    chunk_index=idx,
                    text=text,
                )
            )
    return records

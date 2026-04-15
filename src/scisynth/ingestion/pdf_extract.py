from __future__ import annotations

from pathlib import Path


def _extract_page_text(page) -> str:
    """Extract page text with layout-aware fallback for algorithm/table pages."""
    text = (page.get_text("text") or "").strip()
    # Some scientific PDFs render algorithm blocks poorly in plain text mode.
    # Fall back to block extraction when plain output is too sparse.
    if len(text) >= 200:
        return text
    blocks = page.get_text("blocks") or []
    if not blocks:
        return text
    # blocks: (x0, y0, x1, y1, text, block_no, block_type)
    block_texts: list[str] = []
    for block in sorted(blocks, key=lambda b: (b[1], b[0])):
        b_text = str(block[4] or "").strip()
        if b_text:
            block_texts.append(b_text)
    merged = "\n\n".join(block_texts).strip()
    return merged or text


def text_from_pdf_bytes(data: bytes, *, max_chars: int = 1_500_000) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        parts: list[str] = []
        total = 0
        for page in doc:
            t = _extract_page_text(page)
            if total + len(t) > max_chars:
                parts.append(t[: max(0, max_chars - total)])
                break
            parts.append(t)
            total += len(t)
        return "\n\n".join(parts).strip()
    finally:
        doc.close()


def text_from_pdf_path(path: Path, *, max_chars: int = 1_500_000) -> str:
    """Extract plain text from a PDF file on disk."""
    return text_from_pdf_bytes(path.read_bytes(), max_chars=max_chars)

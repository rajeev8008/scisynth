from __future__ import annotations

from pathlib import Path


def text_from_pdf_bytes(data: bytes, *, max_chars: int = 1_500_000) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        parts: list[str] = []
        total = 0
        for page in doc:
            t = page.get_text("text") or ""
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

from __future__ import annotations

import logging
import re
from dataclasses import replace

import arxiv
import httpx

from scisynth.config import Settings
from scisynth.ingestion.arxiv_cache import arxiv_document_cache
from scisynth.ingestion.arxiv_loader import _map_result_to_document
from scisynth.ingestion.pdf_extract import text_from_pdf_bytes
from scisynth.ingestion.schema import PaperDocument

logger = logging.getLogger(__name__)

# https://arxiv.org/abs/1706.03762 , .../pdf/1706.03762.pdf
_ARXIV_HOST = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^?\s#]+)",
    re.IGNORECASE,
)
# new-style id with optional version
_NEW_ID = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def parse_arxiv_reference(ref: str) -> str:
    """Normalize user input (URL, arxiv:1706.03762, or bare id) to an arXiv id for the API."""
    s = ref.strip()
    if not s:
        raise ValueError("ArXiv reference is empty.")
    m = _ARXIV_HOST.search(s)
    if m:
        part = m.group(1).strip()
        if part.lower().endswith(".pdf"):
            part = part[:-4]
        return part
    if s.lower().startswith("arxiv:"):
        s = s.split(":", 1)[1].strip()
    s = s.strip()
    if _NEW_ID.match(s):
        return s
    if re.match(r"^[\w.-]+/\d{7}(v\d+)?$", s):
        return s
    raise ValueError(
        "Could not parse an arXiv id. Use a link like https://arxiv.org/abs/1706.03762 "
        "or an id like 1706.03762.",
    )


def download_arxiv_pdf_bytes(arxiv_id: str, settings: Settings) -> bytes:
    """Download PDF bytes from arxiv.org (subject to size cap)."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": "scisynth/0.1 (research; +https://github.com/)"}
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
    data = response.content
    if len(data) > settings.arxiv_pdf_max_bytes:
        raise ValueError(
            f"PDF larger than configured limit ({settings.arxiv_pdf_max_bytes} bytes).",
        )
    return data


def enrich_paper_with_pdf_text(doc: PaperDocument, settings: Settings) -> PaperDocument:
    """Replace document text with title + abstract + extracted PDF body when possible."""
    data = download_arxiv_pdf_bytes(doc.paper_id, settings)
    body = text_from_pdf_bytes(data, max_chars=settings.arxiv_pdf_max_extract_chars)
    if not body.strip():
        logger.warning("No text extracted from PDF for %s; using metadata text.", doc.paper_id)
        return doc
    full_text = f"{doc.title}\n\n{doc.abstract}\n\n---\n\n{body}"
    return replace(doc, text=full_text)


def fetch_paper_by_arxiv_id(
    arxiv_id: str,
    *,
    settings: Settings | None = None,
    topic: str = "arxiv-on-demand",
) -> PaperDocument:
    """Fetch one paper: metadata via arXiv API; optionally full PDF text (see Settings)."""
    cache_key = f"{arxiv_id}:{bool(settings and settings.arxiv_fetch_full_pdf)}"
    if settings:
        hit = arxiv_document_cache.get(cache_key)
        if hit is not None:
            return hit

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id], max_results=2)
    results = list(client.results(search))
    if not results:
        raise ValueError(f"No arXiv paper found for id {arxiv_id!r}. Check the id or try again later.")
    if len(results) > 1:
        logger.warning("Multiple results for id %s; using the first match.", arxiv_id)
    doc = _map_result_to_document(results[0], topic)

    if settings and settings.arxiv_fetch_full_pdf:
        try:
            doc = enrich_paper_with_pdf_text(doc, settings)
        except Exception as exc:
            logger.warning(
                "Full PDF unavailable for %s (%s); using title+abstract only.",
                arxiv_id,
                exc,
            )

    if settings:
        arxiv_document_cache.put(cache_key, doc)
    return doc

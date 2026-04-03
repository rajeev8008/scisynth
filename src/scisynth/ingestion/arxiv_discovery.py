from __future__ import annotations

import logging
import re

import arxiv

from scisynth.config import Settings
from scisynth.ingestion.arxiv_loader import _map_result_to_document
from scisynth.ingestion.arxiv_single import enrich_paper_with_pdf_text
from scisynth.ingestion.schema import PaperDocument

logger = logging.getLogger(__name__)


def _sanitize_arxiv_query(q: str) -> str:
    """Keep a reasonable arXiv API query string from a user question."""
    s = re.sub(r"\s+", " ", q.strip())[: 400]
    return s if s else "all"


def search_arxiv_papers(settings: Settings, query: str) -> list[PaperDocument]:
    """Keyword search on arXiv; returns up to ``arxiv_discovery_max_results`` papers."""
    q = _sanitize_arxiv_query(query)
    client = arxiv.Client()
    search = arxiv.Search(
        query=q,
        max_results=settings.arxiv_discovery_max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    docs: list[PaperDocument] = []
    for result in client.results(search):
        doc = _map_result_to_document(result, settings.arxiv_discovery_topic)
        if settings.arxiv_discovery_use_full_pdf:
            try:
                doc = enrich_paper_with_pdf_text(doc, settings)
            except Exception as exc:
                logger.warning("PDF fetch failed for %s: %s", doc.paper_id, exc)
        docs.append(doc)
    docs.sort(key=lambda d: d.paper_id)
    return docs

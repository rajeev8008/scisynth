from __future__ import annotations

import logging
from datetime import datetime

import arxiv

from scisynth.config import Settings
from scisynth.ingestion.schema import PaperDocument

logger = logging.getLogger(__name__)


def load_arxiv_documents(settings: Settings) -> list[PaperDocument]:
    """Fetch papers from arXiv and map them to paper documents.

    Args:
        settings: Application settings containing arXiv query values.
    Returns:
        List of mapped paper documents sorted by paper_id.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=settings.arxiv_query,
        max_results=settings.arxiv_max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    docs: list[PaperDocument] = []
    for result in client.results(search):
        docs.append(_map_result_to_document(result, settings.arxiv_topic))
    docs.sort(key=lambda item: item.paper_id)
    return docs


def _map_result_to_document(result: arxiv.Result, topic: str) -> PaperDocument:
    """Map one arXiv result object into local schema.

    Args:
        result: arXiv API result object.
        topic: Topic label to assign.
    Returns:
        PaperDocument representation for ingestion.
    """
    published_year = _extract_year(result)
    authors = ", ".join(author.name for author in result.authors) or "unknown"
    paper_id = result.get_short_id()
    text = f"{result.title}\n\n{result.summary}"
    return PaperDocument(
        paper_id=paper_id,
        title=result.title,
        authors=authors,
        year=published_year,
        topic=topic,
        abstract=result.summary,
        source_path=result.entry_id,
        text=text,
    )


def _extract_year(result: arxiv.Result) -> int:
    """Extract publication year from arXiv result.

    Args:
        result: arXiv API result object.
    Returns:
        Publication year or 0 when unavailable.
    """
    published = getattr(result, "published", None)
    if isinstance(published, datetime):
        return published.year
    return 0

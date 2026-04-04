"""Researcher node — runs multi-hop RAG retrieval for the current section.

Supports two evidence sources:
  - **index**: searches the ingested local corpus (chunks.jsonl)
  - **arxiv**: searches arXiv on-the-fly for each query, downloads PDFs, chunks in memory

The source is chosen based on the ``research_source`` field in state
(defaults to "arxiv" so deep research works even without ingested papers).
"""

from __future__ import annotations

import logging

from scisynth.agent.multihop import build_hop2_query, evidence_insufficient, merge_chunk_lists
from scisynth.config import get_settings
from scisynth.research.state import ResearchState
from scisynth.retrieval.contract import RetrievedChunk

logger = logging.getLogger(__name__)


def _chunks_to_dicts(chunks: list[RetrievedChunk]) -> list[dict]:
    """Serialise RetrievedChunk objects to plain dicts for state storage."""
    return [
        {
            "chunk_id": c.id,
            "text": c.text,
            "score": c.score,
            "paper_id": c.paper_id or "",
            "paper_title": c.paper_title or "",
        }
        for c in chunks
    ]


def _retrieve_from_index(queries: list[str], settings) -> list[RetrievedChunk]:
    """Retrieve from the ingested local index (BM25/hybrid)."""
    from scisynth.retrieval.factory import get_retriever

    retriever = get_retriever(settings=settings)
    top_k = settings.answer_top_k
    all_chunks: list[RetrievedChunk] = []

    for query in queries:
        if not query.strip():
            continue
        h1 = retriever.retrieve(query.strip(), top_k=top_k)
        if settings.rag_multi_hop and settings.rag_max_hops >= 2 and evidence_insufficient(h1, settings):
            q2 = build_hop2_query(query, h1)
            h2 = retriever.retrieve(q2, top_k=settings.rag_hop2_top_k)
            merged = merge_chunk_lists(h1, h2)
            all_chunks.extend(merged)
        else:
            all_chunks.extend(h1)

    return all_chunks


def _retrieve_from_arxiv(queries: list[str], settings) -> list[RetrievedChunk]:
    """Search arXiv for each query, download PDFs, chunk, and retrieve."""
    from scisynth.ingestion.arxiv_discovery import search_arxiv_papers
    from scisynth.ingestion.schema import ChunkRecord
    from scisynth.ingestion.transform import chunk_documents
    from scisynth.retrieval.memory_bm25 import InMemoryBM25Retriever

    all_chunks: list[RetrievedChunk] = []
    top_k = settings.answer_top_k

    for query in queries:
        if not query.strip():
            continue
        try:
            docs = search_arxiv_papers(settings, query.strip())
        except Exception as exc:
            logger.warning("arXiv search failed for query %r: %s", query[:60], exc)
            continue

        if not docs:
            logger.info("No arXiv results for query: %s", query[:60])
            continue

        # Chunk all fetched papers
        chunks: list[ChunkRecord] = []
        titles: dict[str, str] = {}
        for doc in docs:
            titles[doc.paper_id] = doc.title
            chunks.extend(
                chunk_documents([doc], settings.chunk_size, settings.chunk_overlap),
            )

        if not chunks:
            continue

        # Build ephemeral BM25 index and retrieve
        retriever = InMemoryBM25Retriever(chunks, titles)
        results = retriever.retrieve(query.strip(), top_k=top_k)
        all_chunks.extend(results)

    return all_chunks


def researcher_node(state: ResearchState) -> dict:
    """Retrieve evidence for the current outline section.

    Uses arXiv discovery by default (fetches real papers), or falls back to the
    ingested index when ``research_source`` is set to "index" in state.
    """
    settings = get_settings()
    idx = state["current_section_idx"]
    outline = state.get("outline", [])

    if idx >= len(outline):
        logger.warning("Researcher: section index %d out of range", idx)
        return {}

    section = outline[idx]
    queries = section.get("queries", [section.get("title", "")])
    if not queries:
        queries = [section.get("title", state["topic"])]

    # If the reviewer sent us back with feedback, augment queries with it
    review = state.get("section_reviews", {}).get(str(idx), {})
    reviewer_feedback = review.get("feedback", "")
    if review.get("action") == "research_more" and reviewer_feedback:
        logger.info("Researcher: using reviewer feedback to refine queries: %s", reviewer_feedback[:80])
        # Add the feedback as an additional search query to broaden evidence
        queries = queries + [f"{section.get('title', '')} {reviewer_feedback}"]

    # Choose evidence source
    source = state.get("research_source", "arxiv")

    if source == "index":
        all_chunks = _retrieve_from_index(queries, settings)
    else:
        # Default: arXiv discovery (works without ingested data)
        all_chunks = _retrieve_from_arxiv(queries, settings)
        # If arXiv returns nothing, fall back to index
        if not all_chunks:
            logger.info("arXiv returned no results, falling back to index")
            all_chunks = _retrieve_from_index(queries, settings)

    # Deduplicate by chunk_id, keep best score
    best: dict[str, RetrievedChunk] = {}
    for c in all_chunks:
        prev = best.get(c.id)
        if prev is None or c.score > prev.score:
            best[c.id] = c

    final = sorted(best.values(), key=lambda c: -c.score)[:settings.answer_top_k * 2]

    logger.info(
        "Researcher [%s]: section %d (%s) — %d chunks from %d queries",
        source, idx, section.get("title", "?")[:40], len(final), len(queries),
    )

    return {
        "section_evidence": {str(idx): _chunks_to_dicts(final)},
    }

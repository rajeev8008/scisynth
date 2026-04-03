from __future__ import annotations

from collections.abc import Iterator

from scisynth.agent.llm_client import generate_answer_text, generate_answer_text_stream
from scisynth.agent.models import AnswerResult, Citation
from scisynth.agent.multihop import (
    build_hop2_query,
    evidence_insufficient,
    merge_chunk_lists,
)
from scisynth.agent.prompting import build_answer_prompt
from scisynth.config import Settings
from scisynth.ingestion.schema import ChunkRecord
from scisynth.retrieval.contract import Retriever, RetrievedChunk
from scisynth.retrieval.factory import get_retriever


def answer_question_with_arxiv(
    question: str,
    arxiv_ref: str,
    *,
    settings: Settings,
    top_k: int | None = None,
    temperature: float | None = None,
) -> AnswerResult:
    """Fetch one arXiv paper by URL or id, chunk in memory, then answer with BM25 retrieval."""
    from scisynth.ingestion.arxiv_single import fetch_paper_by_arxiv_id, parse_arxiv_reference
    from scisynth.ingestion.transform import chunk_documents
    from scisynth.retrieval.memory_bm25 import InMemoryBM25Retriever

    arxiv_id = parse_arxiv_reference(arxiv_ref)
    doc = fetch_paper_by_arxiv_id(arxiv_id, settings=settings)
    chunks = chunk_documents(
        [doc],
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if not chunks:
        return AnswerResult(
            question=question,
            answer="No text chunks were produced from this paper (unexpected).",
            citations=[],
            model=settings.llm_model,
            retrieval_hops_used=1,
        )
    titles = {doc.paper_id: doc.title}
    retriever = InMemoryBM25Retriever(chunks, titles)
    return answer_question(
        question,
        settings=settings,
        retriever=retriever,
        top_k=top_k,
        temperature=temperature,
    )


def answer_question_with_arxiv_discovery(
    question: str,
    *,
    settings: Settings,
    top_k: int | None = None,
    temperature: float | None = None,
) -> AnswerResult:
    """Keyword-search arXiv for ``question``, optionally enrich with PDFs, then answer in memory."""
    from scisynth.ingestion.arxiv_discovery import search_arxiv_papers
    from scisynth.ingestion.transform import chunk_documents
    from scisynth.retrieval.memory_bm25 import InMemoryBM25Retriever

    try:
        docs = search_arxiv_papers(settings, question)
    except Exception as exc:
        raise RuntimeError(
            "arXiv search failed (network or API). Try again later or use a single arXiv id.",
        ) from exc
    if not docs:
        return AnswerResult(
            question=question,
            answer="No arXiv papers matched this search. Try different keywords or paste a known arXiv id.",
            citations=[],
            model=settings.llm_model,
            retrieval_hops_used=1,
        )
    chunks: list[ChunkRecord] = []
    titles: dict[str, str] = {}
    for doc in docs:
        titles[doc.paper_id] = doc.title
        chunks.extend(
            chunk_documents([doc], settings.chunk_size, settings.chunk_overlap),
        )
    if not chunks:
        return AnswerResult(
            question=question,
            answer="Papers were found but produced no text chunks (unexpected).",
            citations=[],
            model=settings.llm_model,
            retrieval_hops_used=1,
        )
    retriever = InMemoryBM25Retriever(chunks, titles)
    return answer_question(
        question,
        settings=settings,
        retriever=retriever,
        top_k=top_k,
        temperature=temperature,
    )


def retrieve_chunks_for_answer(
    question: str,
    *,
    settings: Settings,
    retriever: Retriever | None,
    top_k: int | None,
) -> tuple[list[RetrievedChunk], int]:
    """Multi-hop retrieval: ranked chunks and hop count (1 or 2)."""
    effective_top_k = max(1, top_k if top_k is not None else settings.answer_top_k)
    engine = retriever or get_retriever(settings=settings)

    k1 = settings.rag_hop1_top_k if settings.rag_hop1_top_k is not None else effective_top_k
    chunks_h1 = engine.retrieve(question, top_k=k1)
    hops_used = 1

    if not chunks_h1:
        return [], hops_used

    chunks_final: list[RetrievedChunk]
    if (
        settings.rag_multi_hop
        and settings.rag_max_hops >= 2
        and evidence_insufficient(chunks_h1, settings)
    ):
        q2 = build_hop2_query(question, chunks_h1)
        chunks_h2 = engine.retrieve(q2, top_k=settings.rag_hop2_top_k)
        hops_used = 2
        chunks_final = merge_chunk_lists(chunks_h1, chunks_h2)
        chunks_final = sorted(chunks_final, key=lambda c: -c.score)[:effective_top_k]
    else:
        chunks_final = sorted(chunks_h1, key=lambda c: -c.score)[:effective_top_k]

    return chunks_final, hops_used


def answer_question(
    question: str,
    *,
    settings: Settings,
    retriever: Retriever | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
) -> AnswerResult:
    """Run question -> retrieve (multi-hop when configured) -> LLM -> cited answer.

    When ``rag_multi_hop`` is on and first-hop evidence looks weak, a second retrieval
    runs with an expanded query built from top first-hop passages, then chunks are
    merged and deduplicated before generation.
    """
    effective_temp = temperature if temperature is not None else settings.answer_temperature
    chunks_final, hops_used = retrieve_chunks_for_answer(
        question,
        settings=settings,
        retriever=retriever,
        top_k=top_k,
    )

    if not chunks_final:
        return AnswerResult(
            question=question,
            answer="I do not know based on the available indexed context.",
            citations=[],
            model=settings.llm_model,
            retrieval_hops_used=hops_used,
        )

    prompt = build_answer_prompt(
        question,
        chunks_final,
        retrieval_hops_used=hops_used,
    )
    answer = generate_answer_text(
        settings,
        prompt,
        temperature=effective_temp,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer=answer,
        citations=build_citations(chunks_final),
        model=settings.llm_model,
        retrieval_hops_used=hops_used,
    )


def answer_question_stream(
    question: str,
    *,
    settings: Settings,
    retriever: Retriever | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """Same retrieval as ``answer_question``, then stream LLM token deltas.

    Raises:
        ValueError: When retrieval returns no chunks (same abstain case as non-streaming).
    """
    effective_temp = temperature if temperature is not None else settings.answer_temperature
    chunks_final, hops_used = retrieve_chunks_for_answer(
        question,
        settings=settings,
        retriever=retriever,
        top_k=top_k,
    )
    if not chunks_final:
        raise ValueError(
            "No indexed context matched this question. Ingest documents or use arXiv single/discovery.",
        )
    prompt = build_answer_prompt(
        question,
        chunks_final,
        retrieval_hops_used=hops_used,
    )
    yield from generate_answer_text_stream(
        settings,
        prompt,
        temperature=effective_temp,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Convert retrieved chunks into response citation objects.

    Args:
        chunks: Retrieved chunk list.
    Returns:
        Citation list preserving chunk rank order.
    """
    out: list[Citation] = []
    for chunk in chunks:
        pid = chunk.paper_id or _extract_paper_id(chunk.id)
        out.append(
            Citation(
                chunk_id=chunk.id,
                paper_id=pid,
                paper_title=chunk.paper_title,
                snippet=_snippet(chunk.text),
                score=chunk.score,
            )
        )
    return out


def _extract_paper_id(chunk_id: str) -> str:
    """Extract paper identifier from a chunk id pattern.

    Args:
        chunk_id: Retrieved chunk identifier.
    Returns:
        Parsed paper id prefix or the original chunk id.
    """
    marker = ":chunk-"
    if marker not in chunk_id:
        return chunk_id
    return chunk_id.split(marker, maxsplit=1)[0]


def _snippet(text: str, width: int = 180) -> str:
    """Build a short single-line snippet from chunk text.

    Args:
        text: Full chunk text.
        width: Maximum characters to include.
    Returns:
        Trimmed snippet string.
    """
    normalized = " ".join(text.split())
    if len(normalized) <= width:
        return normalized
    return normalized[: width - 3].rstrip() + "..."

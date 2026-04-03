from __future__ import annotations

from scisynth.agent.llm_client import generate_answer_text
from scisynth.agent.models import AnswerResult, Citation
from scisynth.agent.prompting import build_answer_prompt
from scisynth.config import Settings
from scisynth.retrieval.contract import Retriever, RetrievedChunk
from scisynth.retrieval.factory import get_retriever


def answer_question(
    question: str,
    *,
    settings: Settings,
    retriever: Retriever | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
) -> AnswerResult:
    """Run question -> retrieve -> LLM and return answer with citations.

    Args:
        question: End-user question.
        settings: Runtime settings.
        retriever: Optional retriever override for testing.
        top_k: Optional retrieval override.
        temperature: Optional model temperature override.
    Returns:
        Answer result containing answer text and citations.
    """
    effective_top_k = max(1, top_k if top_k is not None else settings.answer_top_k)
    effective_temp = temperature if temperature is not None else settings.answer_temperature
    engine = retriever or get_retriever(settings=settings)
    chunks = engine.retrieve(question, top_k=effective_top_k)
    if not chunks:
        return AnswerResult(
            question=question,
            answer="I do not know based on the available indexed context.",
            citations=[],
            model=settings.llm_model,
        )
    prompt = build_answer_prompt(question, chunks)
    answer = generate_answer_text(
        settings,
        prompt,
        temperature=effective_temp,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer=answer,
        citations=_build_citations(chunks),
        model=settings.llm_model,
    )


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Convert retrieved chunks into response citation objects.

    Args:
        chunks: Retrieved chunk list.
    Returns:
        Citation list preserving chunk rank order.
    """
    out: list[Citation] = []
    for chunk in chunks:
        out.append(
            Citation(
                chunk_id=chunk.id,
                paper_id=_extract_paper_id(chunk.id),
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

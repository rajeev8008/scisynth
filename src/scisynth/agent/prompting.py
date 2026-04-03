from __future__ import annotations

from scisynth.retrieval.contract import RetrievedChunk


def build_answer_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    retrieval_hops_used: int = 1,
) -> str:
    """Build the answer-generation prompt with grounded evidence.

    Args:
        question: End-user natural language question.
        chunks: Retrieved context chunks ranked by relevance.
        retrieval_hops_used: 2 when a second retrieval pass augmented evidence.
    Returns:
        Prompt string for a chat completion model.
    """
    context = _format_context(chunks)
    hop_note = ""
    if retrieval_hops_used >= 2:
        hop_note = (
            "Context was gathered with two retrieval passes; "
            "the second pass used expanded query text from the first-hit passages.\n"
        )
    return (
        "You are a scientific research assistant.\n"
        "Answer only using the provided context.\n"
        "If the context is insufficient, say you do not know.\n"
        "Cite chunk IDs in square brackets like [chunk_id]. "
        "When a paper title is given, mention it once if helpful.\n"
        f"{hop_note}\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks in a deterministic prompt block.

    Args:
        chunks: Retrieved chunk list.
    Returns:
        String representation of context evidence.
    """
    if not chunks:
        return "(no context)"
    lines: list[str] = []
    for chunk in chunks:
        title = chunk.paper_title or chunk.paper_id or "unknown"
        lines.append(
            f"[{chunk.id}] paper={chunk.paper_id} title={title!r} score={chunk.score:.3f}"
        )
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()

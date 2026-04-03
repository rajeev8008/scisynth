from __future__ import annotations

from scisynth.retrieval.contract import RetrievedChunk


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the answer-generation prompt with grounded evidence.

    Args:
        question: End-user natural language question.
        chunks: Retrieved context chunks ranked by relevance.
    Returns:
        Prompt string for a chat completion model.
    """
    context = _format_context(chunks)
    return (
        "You are a scientific research assistant.\n"
        "Answer only using the provided context.\n"
        "If the context is insufficient, say you do not know.\n"
        "Cite chunk IDs in square brackets like [paper:chunk-0].\n\n"
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
        lines.append(f"[{chunk.id}] score={chunk.score:.3f}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()

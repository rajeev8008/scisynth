from __future__ import annotations

import re

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
            "Note: Context was gathered with two retrieval passes for broader coverage.\n"
        )
    return (
        "You are a scientific research assistant. "
        "Answer the question using ONLY the provided context passages.\n\n"
        "FORMATTING RULES (follow strictly):\n"
        "- Write in clear, readable prose. Do NOT include chunk IDs, paper IDs, "
        "arXiv identifiers (like 1706.03762v7), or any internal reference codes in your answer.\n"
        "- When referencing a paper, mention it by its TITLE in natural language "
        "(e.g., 'as described in the paper \"Attention Is All You Need\"').\n"
        "- For mathematical equations, use LaTeX notation wrapped in dollar signs: "
        "$inline$ for inline math, $$block$$ for display equations. "
        "Reconstruct equations properly from the context even if the raw text is garbled.\n"
        "- Use Markdown formatting: headings (##), bold, bullet points, and numbered lists for clarity.\n"
        "- If the context is insufficient, say so clearly.\n"
        f"{hop_note}\n"
        f"**Question:**\n{question}\n\n"
        f"**Context Passages:**\n{context}\n\n"
        "**Answer:**"
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks in a clean prompt block.

    Args:
        chunks: Retrieved chunk list.
    Returns:
        String representation of context evidence.
    """
    if not chunks:
        return "(no context)"
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.paper_title or "Unknown Paper"
        cleaned_text = _clean_pdf_text(chunk.text)
        lines.append(f"--- Passage {i} (from: {title}) ---")
        lines.append(cleaned_text)
        lines.append("")
    return "\n".join(lines).strip()


def _clean_pdf_text(text: str) -> str:
    """Clean up raw PDF-extracted text for better readability.

    Fixes common issues from PyMuPDF extraction: broken line breaks in the
    middle of sentences, excessive whitespace, stray special characters, etc.
    """
    # Replace single newlines that break mid-sentence with spaces
    # (keep double newlines as paragraph breaks)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Fix common PDF extraction artifacts: '- ' at line breaks (hyphenation)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    # Normalize unicode dashes/minuses to standard ASCII
    text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '--')
    # Remove stray form-feed, vertical tab
    text = text.replace('\f', '').replace('\v', '')
    return text.strip()

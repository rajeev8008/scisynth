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
    formatted_chunks = _format_context(chunks)
    return (
        "You are an expert scientific research assistant. Your task is to answer the user's question based strictly on the provided Context from a specific academic paper.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. You MUST formulate your answer using ONLY the facts, data, and claims present in the Context below.\n"
        "2. DO NOT use your pre-existing knowledge or external facts, even if you know them to be true.\n"
        '3. If the provided Context does not contain sufficient information to answer the question, you must explicitly output exactly this sentence: "The retrieved sections of this paper do not contain enough information to answer your question."\n'
        "4. Write in clean, reader-friendly Markdown with short headings/bullets where useful.\n"
        "5. For equations, use proper LaTeX only: inline `$...$` and display `$$...$$`.\n"
        "6. Do NOT use square-bracket math blocks like `[ ... ]`.\n"
        "7. Do NOT mention passage numbers, chunk IDs, or labels like 'Passage 3'.\n"
        "8. If you cite support, reference it naturally (e.g., 'the paper states...'), not by passage labels.\n\n"
        f"Context:\n{formatted_chunks}\n\n"
        f"User Question: {question}"
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

"""Prompt templates for every agent node in the deep research pipeline.

Each function returns a plain string prompt ready for the LLM.
Prompts are optimised for smaller instruct models (llama-3.1-8b-instant, etc.)
by being concise and explicit about the expected output format.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def planner_prompt(topic: str, *, max_sections: int = 5) -> str:
    return (
        "You are a research planning assistant.\n"
        f"Given a research topic, break it into {max_sections} focused sections for a literature review.\n\n"
        "Return ONLY a JSON array. Each element:\n"
        '  {"title": "Section Title", "description": "1-2 sentence scope", '
        '"queries": ["search query 1", "search query 2"]}\n\n'
        "Rules:\n"
        "- Each section should cover a distinct sub-topic\n"
        "- Queries should be specific enough to find relevant scientific papers\n"
        "- Do NOT include introduction or conclusion sections\n"
        f"- Return exactly {max_sections} sections\n\n"
        f"Topic: {topic}\n\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def writer_prompt(
    section_title: str,
    section_description: str,
    evidence_text: str,
) -> str:
    return (
        "You are a scientific research writer.\n"
        "Write a detailed section for a research report using ONLY the provided evidence.\n\n"
        "FORMATTING RULES (follow strictly):\n"
        "- Reference papers by their TITLE in natural language "
        '(e.g., \'as shown in "Attention Is All You Need"\')\n'
        "- Do NOT include chunk IDs, paper IDs, arXiv identifiers, or any internal "
        "reference codes like [chunk_id] or [1706.03762v7] in the text\n"
        "- For mathematical equations, use LaTeX notation: $inline$ for inline, "
        "$$block$$ for display equations. Reconstruct equations properly even if "
        "the raw evidence text is garbled\n"
        "- Write 2-4 substantive paragraphs with clear Markdown formatting\n"
        "- Be specific about methods, findings, and implications\n"
        "- If evidence is insufficient, note the gap but do NOT hallucinate\n\n"
        f"## Section: {section_title}\n"
        f"**Scope:** {section_description}\n\n"
        f"### Evidence:\n{evidence_text}\n\n"
        f"### Write the section '{section_title}':\n"
    )


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------

def reviewer_prompt(
    section_title: str,
    section_description: str,
    draft: str,
    evidence_text: str,
) -> str:
    return (
        "You are a research quality reviewer.\n"
        "Evaluate the draft section against the evidence and criteria.\n\n"
        "Return ONLY valid JSON:\n"
        '{"passed": true/false, "feedback": "brief feedback", '
        '"action": "accept" | "research_more" | "rewrite"}\n\n'
        "Action meanings:\n"
        '- "accept": draft is good enough\n'
        '- "research_more": need more/different evidence (triggers new retrieval)\n'
        '- "rewrite": evidence is fine but draft needs improvement\n\n'
        "Criteria:\n"
        "1. Does the draft address the section topic adequately?\n"
        "2. Are there at least 2 references to source papers (by title, not IDs)?\n"
        "3. Is content grounded in the evidence (no hallucination)?\n"
        "4. Is the writing clear and academic?\n"
        "5. Are equations properly formatted in LaTeX ($...$ or $$...$$)?\n"
        "6. Does the text avoid raw chunk IDs, paper IDs, or arXiv identifiers?\n\n"
        f"Section: {section_title}\n"
        f"Scope: {section_description}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def synthesizer_prompt(topic: str, sections_text: str) -> str:
    return (
        "You are a research report synthesizer.\n"
        "Merge the following section drafts into a single cohesive research report.\n\n"
        "FORMATTING RULES (follow strictly):\n"
        "- Add a brief Introduction (2-3 sentences framing the topic)\n"
        "- Include all section content, referencing papers by TITLE only\n"
        "- Add a Conclusion (3-5 sentences summarising key findings and open questions)\n"
        "- Use Markdown formatting with ## headings\n"
        "- For equations, use LaTeX: $inline$ or $$display$$\n"
        "- Do NOT include chunk IDs, paper IDs, arXiv identifiers, or internal codes\n"
        "- Make smooth transitions between sections\n"
        "- End with a References section listing paper titles mentioned\n\n"
        f"# Research Topic: {topic}\n\n"
        f"## Section Drafts:\n{sections_text}\n\n"
        "## Write the complete, cohesive report:\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_evidence_text(text: str) -> str:
    """Clean raw PDF-extracted text for evidence display."""
    # Replace single newlines mid-sentence with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Fix hyphenation artifacts
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    # Normalize unicode dashes
    text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '--')
    # Remove stray form-feed, vertical tab
    text = text.replace('\f', '').replace('\v', '')
    return text.strip()


def format_evidence_for_prompt(chunks: list[dict], *, max_chars: int = 6000) -> str:
    """Format retrieved chunk dicts into a prompt-ready evidence block."""
    if not chunks:
        return "(no evidence retrieved)"
    lines: list[str] = []
    total = 0
    for i, c in enumerate(chunks, 1):
        paper = c.get("paper_title") or "Unknown Paper"
        text = _clean_evidence_text(c.get("text", ""))
        entry = f"--- Passage {i} (from: {paper}) ---\n{text}\n"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines).strip() if lines else "(no evidence retrieved)"

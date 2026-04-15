"""Prompt templates for every agent node in the deep research pipeline.

Each function returns a plain string prompt ready for the LLM.
"""

from __future__ import annotations

import re

_MATH_FORMATTING_RULES = (
    "MATH FORMAT (strict):\n"
    "- Use only valid LaTeX.\n"
    "- Inline math MUST use $...$.\n"
    "- Block math MUST use $$...$$.\n"
    "- Do NOT use \\(...\\), \\[...\\], bare [ ... ], bare ( ... ), or malformed separators like ;=;.\n"
    "- Use clean LaTeX commands such as \\frac{}{}, \\pi_{\\theta}, and \\mathcal{L} where relevant.\n"
    "- Do NOT output raw artifacts like !\\bigl, ,, or broken escape fragments.\n"
    "- Split long equations into separate blocks and add a short explanation below each equation.\n"
    "- If an equation looks malformed, rewrite it into clean, renderable LaTeX before returning.\n"
)


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
# Researcher (Dynamic Search)
# ---------------------------------------------------------------------------


def researcher_search_prompt(topic: str, current_section: str, feedback: str = "") -> str:
    """Generates a dynamic ArXiv boolean search string based on the section topic."""
    feedback_text = f"\nPrevious Search Feedback: {feedback}\n" if feedback else "\n"
    return (
        "You are an expert academic librarian. Your task is to convert a specific section topic "
        "from a research outline into a highly optimized boolean search query for the ArXiv academic database.\n\n"
        f"Report Topic: {topic}\n"
        f"Current Section to Research: {current_section}\n"
        f"{feedback_text}"
        "INSTRUCTIONS:\n"
        "1. Identify the core keywords from the Current Section.\n"
        "2. Format the output as a strict boolean search string using AND/OR operators.\n"
        "3. Do NOT restrict the search to specific ArXiv categories unless absolutely necessary. Search all text fields using 'all:' (e.g., all:\"keyword\").\n"
        "4. Output ONLY the raw search string. Do not include quotes around the final output, greetings, or explanations.\n\n"
        "Example Output:\n"
        "all:\"microplastic\" AND all:\"freshwater\" AND (all:\"mitigation\" OR all:\"removal\")\n\n"
        "Query:"
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
        "You are drafting a specific section for a comprehensive academic report.\n\n"
        f"Section Topic: {section_title}\n"
        f"Section Scope: {section_description}\n\n"
        "Evidence Retrieved by Researcher:\n"
        f"{evidence_text}\n\n"
        "CRITICAL RULES FOR WRITING:\n"
        "1. NO HALLUCINATIONS: You may ONLY synthesize facts, statistics, and claims explicitly stated in the Evidence above.\n"
        "2. NO OUTSIDE KNOWLEDGE: Under no circumstances should you use your pre-existing knowledge to fill in gaps.\n"
        "3. THE NULL STATE: Read the Evidence carefully. If the Evidence is completely irrelevant to the Section Topic, or if it does not contain enough substantive data to write a meaningful academic paragraph, DO NOT write a generic summary. Instead, output EXACTLY this string and nothing else:\n"
        "INSUFFICIENT_EVIDENCE\n\n"
        f"{_MATH_FORMATTING_RULES}"
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
        "You are a strict Senior Academic Editor reviewing a draft section for a research report.\n\n"
        "Return ONLY valid JSON:\n"
        '{"passed": true/false, "feedback": "Detailed explanation of what needs to be fixed or what new keywords to search for", '
        '"action": "accept" | "research_more" | "rewrite"}\n\n'
        f"Section Topic: {section_title}\n"
        f"Section Scope: {section_description}\n\n"
        f"Evidence Provided to Writer:\n{evidence_text}\n\n"
        f"Draft Written:\n{draft}\n\n"
        "YOUR TASK & CRITERIA:\n"
        "1. If the Draft Written is exactly \"INSUFFICIENT_EVIDENCE\", you MUST output a \"research_more\" action, set passed to false, and instruct the researcher to use different keywords in the feedback.\n"
        "2. If a draft is written, verify that EVERY claim in the Draft is backed up by the Evidence. If the writer hallucinated or added outside facts, output a \"rewrite\" action.\n"
        "3. If the draft is well-written, academic, and strictly grounded ONLY in the Evidence, output an \"accept\" action.\n"
        "4. Are equations properly formatted in LaTeX ($...$ or $$...$$)?\n"
        "5. Does the text avoid raw chunk IDs, paper IDs, or arXiv identifiers?\n"
        "6. Reject malformed math, including bracket-delimited equations, random artifacts (!\\bigl, ,,), or broken spacing/separators.\n\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def synthesizer_intro_outro_prompt(topic: str, sections_text: str) -> str:
    return (
        "You are a research report synthesizer.\n"
        "I have already drafted the body sections for a research report. "
        "Your task is ONLY to write an Introduction and a Conclusion to wrap these sections.\n\n"
        "FORMATTING RULES (follow strictly):\n"
        "- The Introduction should be 2-3 paragraphs framing the topic and outlining the report.\n"
        "- The Conclusion should be 2-3 paragraphs summarising key findings and open questions.\n"
        "- Use clear Markdown formatting if needed.\n"
        "- Do NOT use chunk IDs, paper IDs, arXiv identifiers, or internal codes.\n"
        f"- {_MATH_FORMATTING_RULES}"
        "- Return ONLY valid JSON with two string fields: 'introduction' and 'conclusion'.\n\n"
        f"# Research Topic: {topic}\n\n"
        f"## Body Section Drafts (provided for context only):\n{sections_text}\n\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_evidence_text(text: str) -> str:
    """Clean raw PDF-extracted text for evidence display."""
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '--')
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

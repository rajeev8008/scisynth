"""Synthesizer node — merges all section drafts into a cohesive research report."""

from __future__ import annotations

import logging

from scisynth.agent.llm_client import generate_answer_text
from scisynth.config import get_settings
from scisynth.research.prompts import synthesizer_prompt
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)


def synthesizer_node(state: ResearchState) -> dict:
    """Merge completed section drafts into a final report.

    Returns:
        State update with the final_report text.
    """
    settings = get_settings()
    topic = state["topic"]
    outline = state.get("outline", [])
    drafts = state.get("section_drafts", {})

    # Build sections text from drafts in order
    parts: list[str] = []
    for i, section in enumerate(outline):
        title = section.get("title", f"Section {i + 1}")
        draft = drafts.get(str(i), "_No draft generated for this section._")
        parts.append(f"### {title}\n\n{draft}")

    sections_text = "\n\n---\n\n".join(parts)

    prompt = synthesizer_prompt(topic, sections_text)
    report = generate_answer_text(
        settings,
        prompt,
        temperature=0.25,
        max_output_tokens=settings.llm_max_output_tokens * 3,  # longer output for full report
    )

    logger.info("Synthesizer: produced %d-char report for %r", len(report), topic[:60])

    return {
        "final_report": report,
    }


def advance_section_node(state: ResearchState) -> dict:
    """Move to the next section in the outline and reset iteration count.

    Returns:
        State update advancing the section index.
    """
    new_idx = state["current_section_idx"] + 1
    logger.info("Advancing to section %d", new_idx)
    return {
        "current_section_idx": new_idx,
        "iteration_count": 0,
    }

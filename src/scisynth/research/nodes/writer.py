"""Writer node — drafts a section from evidence chunks."""

from __future__ import annotations

import logging

from scisynth.agent.llm_client import generate_answer_text
from scisynth.config import get_settings
from scisynth.research.prompts import format_evidence_for_prompt, writer_prompt
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)


def writer_node(state: ResearchState) -> dict:
    """Draft the current section using retrieved evidence.

    Returns:
        State update with the section draft text.
    """
    settings = get_settings()
    idx = state["current_section_idx"]
    outline = state.get("outline", [])

    if idx >= len(outline):
        return {}

    section = outline[idx]
    evidence = state.get("section_evidence", {}).get(str(idx), [])

    evidence_text = format_evidence_for_prompt(evidence)
    prompt = writer_prompt(
        section_title=section["title"],
        section_description=section.get("description", ""),
        evidence_text=evidence_text,
    )

    draft = generate_answer_text(
        settings,
        prompt,
        temperature=0.3,
        max_output_tokens=settings.llm_max_output_tokens * 2,  # allow longer output for sections
    )

    logger.info(
        "Writer: section %d (%s) — %d chars draft",
        idx, section.get("title", "?")[:40], len(draft),
    )

    return {
        "section_drafts": {str(idx): draft},
    }

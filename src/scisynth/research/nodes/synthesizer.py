"""Synthesizer node — merges all section drafts into a cohesive research report."""

from __future__ import annotations

import json
import logging
import re

from scisynth.agent.llm_client import generate_answer_text
from scisynth.config import get_settings
from scisynth.research.prompts import synthesizer_intro_outro_prompt
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)


def _parse_intro_outro_json(text: str) -> dict:
    """Parse the synthesizer's JSON response with fallbacks."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Heuristic fallback for malformed JSON, common with smaller models
    intro_match = re.search(r'"introduction"\s*:\s*"?(.*?)"?(?:,\s*"conclusion"|\s*\})', text, re.DOTALL | re.IGNORECASE)
    conc_match = re.search(r'"conclusion"\s*:\s*"?(.*?)"?(?:\s*\}|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if intro_match or conc_match:
        intro_text = intro_match.group(1).replace('\\"', '"').replace('\\n', '\n').strip() if intro_match else "_(Synthesizer failed to format introduction)_"
        conc_text = conc_match.group(1).replace('\\"', '"').replace('\\n', '\n').strip() if conc_match else "_(Synthesizer failed to format conclusion)_"
        return {
            "introduction": intro_text,
            "conclusion": conc_text
        }

    # Fallback if the LLM completely bungles JSON output
    return {
        "introduction": "_(Synthesizer failed to format introduction)_",
        "conclusion": "_(Synthesizer failed to format conclusion)_"
    }


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
        parts.append(f"## {title}\n\n{draft}")

    sections_text = "\n\n---\n\n".join(parts)

    prompt = synthesizer_intro_outro_prompt(topic, sections_text)
    
    try:
        response = generate_answer_text(
            settings,
            prompt,
            temperature=0.25,
            max_output_tokens=1500,  # Intro + Conclusion shouldn't exceed this
        )
        data = _parse_intro_outro_json(response)
    except Exception as exc:
        logger.warning(
            "Synthesizer: LLM call for intro/outro failed (%s). Continuing with body only.", exc
        )
        data = {"introduction": "", "conclusion": ""}

    intro = data.get("introduction", "*(No introduction generated)*").strip()
    conclusion = data.get("conclusion", "*(No conclusion generated)*").strip()

    # Stitch it all together programmatically
    final_report = f"{intro}\n\n---\n\n{sections_text}\n\n---\n\n## Conclusion\n\n{conclusion}"

    logger.info("Synthesizer: produced %d-char report for %r", len(final_report), topic[:60])

    return {
        "final_report": final_report,
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

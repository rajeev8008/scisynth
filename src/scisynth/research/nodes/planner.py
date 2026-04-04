"""Planner node — decomposes a research topic into an outline of sections."""

from __future__ import annotations

import json
import logging
import re

from scisynth.agent.llm_client import generate_answer_text
from scisynth.config import get_settings
from scisynth.research.prompts import planner_prompt
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, tolerant of markdown fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse planner JSON, using fallback outline")
    return []


def _validate_outline(sections: list[dict]) -> list[dict]:
    """Ensure each section has required fields."""
    valid = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        title = s.get("title", "").strip()
        if not title:
            continue
        valid.append({
            "title": title,
            "description": s.get("description", title),
            "queries": s.get("queries", [title]),
        })
    return valid


def planner_node(state: ResearchState) -> dict:
    """Break the topic into a structured research outline.

    Returns:
        State update with outline and reset section index.
    """
    settings = get_settings()
    topic = state["topic"]
    max_sections = min(6, max(3, settings.research_max_sections))

    prompt = planner_prompt(topic, max_sections=max_sections)
    response = generate_answer_text(
        settings,
        prompt,
        temperature=0.4,
        max_output_tokens=1200,
    )

    sections = _validate_outline(_parse_json_array(response))

    # Fallback: if parsing completely failed, create a single-section outline
    if not sections:
        sections = [
            {
                "title": "Overview",
                "description": f"General overview of: {topic}",
                "queries": [topic],
            }
        ]
        logger.warning("Planner fallback: single-section outline for %r", topic)

    logger.info("Planner produced %d sections for topic %r", len(sections), topic[:80])
    return {
        "outline": sections,
        "current_section_idx": 0,
        "iteration_count": 0,
    }

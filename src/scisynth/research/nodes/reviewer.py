"""Reviewer node — evaluates section drafts and decides the next action."""

from __future__ import annotations

import json
import logging
import re

from scisynth.agent.llm_client import generate_answer_text
from scisynth.config import get_settings
from scisynth.research.prompts import format_evidence_for_prompt, reviewer_prompt
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"accept", "research_more", "rewrite"}


def _parse_review_json(text: str) -> dict:
    """Parse the reviewer's JSON response with fallbacks."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Heuristic: if "accept" or "pass" appears, accept
    lower = text.lower()
    if any(w in lower for w in ("accept", "passed", "good", "adequate")):
        return {"passed": True, "feedback": "Accepted", "action": "accept"}

    return {"passed": False, "feedback": "Could not parse review", "action": "accept"}


def reviewer_node(state: ResearchState) -> dict:
    """Evaluate the current section draft and decide next action.

    Returns:
        State update with review result and (potentially) incremented iteration count.
    """
    settings = get_settings()
    idx = state["current_section_idx"]
    outline = state.get("outline", [])

    if idx >= len(outline):
        return {}

    section = outline[idx]
    draft = state.get("section_drafts", {}).get(str(idx), "")
    evidence = state.get("section_evidence", {}).get(str(idx), [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 2)

    # If we've hit the max, force accept
    if iteration_count >= max_iterations:
        logger.info("Reviewer: force-accepting section %d after %d iterations", idx, iteration_count)
        return {
            "section_reviews": {
                str(idx): {
                    "passed": True,
                    "feedback": f"Force-accepted after {iteration_count} iterations",
                    "action": "accept",
                },
            },
        }

    if draft.strip() == "INSUFFICIENT_EVIDENCE":
        logger.info(
            "Reviewer: section %d flagged insufficient evidence; requesting new search",
            idx,
        )
        return {
            "section_reviews": {
                str(idx): {
                    "passed": False,
                    "feedback": "The previous search yielded irrelevant papers. The Writer found insufficient evidence. You MUST generate completely different search keywords for this section.",
                    "action": "research_more",
                },
            },
            "iteration_count": iteration_count + 1,
        }

    evidence_text = format_evidence_for_prompt(evidence)
    prompt = reviewer_prompt(
        section_title=section["title"],
        section_description=section.get("description", ""),
        draft=draft,
        evidence_text=evidence_text,
    )

    try:
        response = generate_answer_text(
            settings,
            prompt,
            temperature=0.2,
            max_output_tokens=300,
        )
        review = _parse_review_json(response)
    except Exception as exc:
        logger.warning(
            "Reviewer: LLM call failed for section %d (%s); auto-accepting draft.",
            idx, exc,
        )
        return {
            "section_reviews": {
                str(idx): {
                    "passed": True,
                    "feedback": "Auto-accepted (reviewer LLM call failed)",
                    "action": "accept",
                },
            },
        }

    action = review.get("action", "accept")
    if action not in _VALID_ACTIONS:
        action = "accept"
    review["action"] = action

    new_count = iteration_count + 1 if action != "accept" else iteration_count

    logger.info(
        "Reviewer: section %d — action=%s passed=%s (iteration %d/%d)",
        idx, action, review.get("passed"), new_count, max_iterations,
    )

    return {
        "section_reviews": {str(idx): review},
        "iteration_count": new_count,
    }

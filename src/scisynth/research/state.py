"""LangGraph state schema for the multi-agent deep research pipeline."""

from __future__ import annotations

from typing import Annotated, TypedDict


def _merge_dicts(old: dict, new: dict) -> dict:
    """Reducer: merge new keys into existing dict (shallow)."""
    if old is None:
        return new or {}
    if new is None:
        return old
    return {**old, **new}


class ResearchState(TypedDict):
    """Full state carried through the LangGraph research graph.

    Fields updated by each node:
        planner     → outline, current_section_idx
        researcher  → section_evidence
        writer      → section_drafts
        reviewer    → section_reviews, iteration_count
        advance     → current_section_idx, iteration_count
        synthesizer → final_report
    """

    topic: str
    outline: list  # [{title, description, queries}]
    current_section_idx: int
    section_evidence: Annotated[dict, _merge_dicts]  # {"0": [chunk_dicts]}
    section_drafts: Annotated[dict, _merge_dicts]  # {"0": "text"}
    section_reviews: Annotated[dict, _merge_dicts]  # {"0": {passed,feedback,action}}
    final_report: str
    iteration_count: int
    max_iterations: int
    research_source: str  # "arxiv" (default) or "index"


def make_initial_state(
    topic: str,
    *,
    max_iterations: int = 2,
    research_source: str = "arxiv",
) -> ResearchState:
    """Create the starting state for a research run."""
    return ResearchState(
        topic=topic,
        outline=[],
        current_section_idx=0,
        section_evidence={},
        section_drafts={},
        section_reviews={},
        final_report="",
        iteration_count=0,
        max_iterations=max_iterations,
        research_source=research_source,
    )

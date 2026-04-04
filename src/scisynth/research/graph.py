"""LangGraph cyclic graph for the multi-agent deep research pipeline.

Graph topology:
    planner → researcher → writer → reviewer ─┐
                 ↑            ↑                │
                 │            │                ▼
                 │            └── (rewrite) ◄─ conditional
                 └── (research_more) ◄────────┘
                                               │
                                          (accept)
                                               ▼
                                        advance_section
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                      ▼
                              researcher           synthesizer → END
                          (next section)          (all sections done)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from scisynth.research.nodes import (
    advance_section_node,
    planner_node,
    researcher_node,
    reviewer_node,
    synthesizer_node,
    writer_node,
)
from scisynth.research.state import ResearchState

logger = logging.getLogger(__name__)


def _review_router(state: ResearchState) -> str:
    """Route after reviewer: loop back or advance."""
    idx = str(state["current_section_idx"])
    review = state.get("section_reviews", {}).get(idx, {})
    action = review.get("action", "accept")

    if action == "research_more":
        return "research_more"
    if action == "rewrite":
        return "rewrite"
    return "accept"


def _sections_router(state: ResearchState) -> str:
    """Route after advancing: more sections or synthesize."""
    idx = state["current_section_idx"]
    outline = state.get("outline", [])
    if idx >= len(outline):
        return "all_done"
    return "next_section"


def build_research_graph() -> StateGraph:
    """Construct and compile the deep research LangGraph.

    Returns:
        Compiled LangGraph ready for .invoke() or .stream().
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("advance_section", advance_section_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("planner")

    # Linear edges
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")

    # Conditional: reviewer decides next action
    graph.add_conditional_edges(
        "reviewer",
        _review_router,
        {
            "research_more": "researcher",
            "rewrite": "writer",
            "accept": "advance_section",
        },
    )

    # Conditional: check if more sections remain
    graph.add_conditional_edges(
        "advance_section",
        _sections_router,
        {
            "next_section": "researcher",
            "all_done": "synthesizer",
        },
    )

    # Synthesizer is terminal
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_research_sync(
    topic: str,
    *,
    max_iterations: int = 2,
    research_source: str = "arxiv",
) -> dict:
    """Run the full research pipeline synchronously.

    Args:
        topic: Research topic or question.
        max_iterations: Max review-loop iterations per section.
        research_source: 'arxiv' (fetch live papers) or 'index' (ingested corpus).

    Returns:
        Final state dict with the completed report.
    """
    from scisynth.research.state import make_initial_state

    graph = build_research_graph()
    initial = make_initial_state(
        topic,
        max_iterations=max_iterations,
        research_source=research_source,
    )

    logger.info("Starting deep research for: %s", topic[:100])
    result = graph.invoke(initial)
    logger.info("Deep research complete: %d chars report", len(result.get("final_report", "")))
    return result


def stream_research(
    topic: str,
    *,
    max_iterations: int = 2,
    research_source: str = "arxiv",
):
    """Stream research pipeline node-by-node.

    Yields:
        (node_name, state_update) tuples after each node completes.
    """
    from scisynth.research.state import make_initial_state

    graph = build_research_graph()
    initial = make_initial_state(
        topic,
        max_iterations=max_iterations,
        research_source=research_source,
    )

    logger.info("Starting streamed deep research for: %s", topic[:100])
    for event in graph.stream(initial):
        for node_name, state_update in event.items():
            yield node_name, state_update

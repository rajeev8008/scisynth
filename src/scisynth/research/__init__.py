"""Multi-agent deep research pipeline — LangGraph orchestration."""

from scisynth.research.graph import build_research_graph, run_research_sync, stream_research
from scisynth.research.models import ReportSection, ResearchCitation, ResearchReport
from scisynth.research.state import ResearchState, make_initial_state

__all__ = [
    "ResearchState",
    "ResearchReport",
    "ReportSection",
    "ResearchCitation",
    "make_initial_state",
    "build_research_graph",
    "run_research_sync",
    "stream_research",
]

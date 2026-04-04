"""Data models for deep research pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResearchCitation:
    """One evidence citation used in a research report section."""

    chunk_id: str
    paper_id: str
    paper_title: str | None
    snippet: str
    score: float


@dataclass(slots=True)
class ReportSection:
    """One section of a research report."""

    title: str
    content: str
    citations: list[ResearchCitation] = field(default_factory=list)
    research_iterations: int = 1


@dataclass(slots=True)
class ResearchReport:
    """Complete deep research report."""

    topic: str
    sections: list[ReportSection] = field(default_factory=list)
    final_report: str = ""
    citations: list[ResearchCitation] = field(default_factory=list)
    model: str = ""
    total_retrieval_hops: int = 0
    total_time_ms: float = 0.0

"""Orchestration, tools, and LLM calls. Phase 4 — keep separate from ingestion."""

from scisynth.agent.models import AnswerResult, Citation
from scisynth.agent.service import (
    answer_question,
    answer_question_stream,
    answer_question_with_arxiv,
    answer_question_with_arxiv_discovery,
)

__all__ = [
    "AnswerResult",
    "Citation",
    "answer_question",
    "answer_question_stream",
    "answer_question_with_arxiv",
    "answer_question_with_arxiv_discovery",
]

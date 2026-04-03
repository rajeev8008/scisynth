from __future__ import annotations

import logging

from scisynth.config import Settings, get_settings
from scisynth.retrieval.contract import Retriever
from scisynth.retrieval.live import LiveRetriever
from scisynth.retrieval.mock import MockRetriever

logger = logging.getLogger(__name__)


def get_retriever(settings: Settings | None = None) -> Retriever:
    """Single factory for retrieval. Do not instantiate retrievers elsewhere."""
    cfg = settings or get_settings()
    mode = cfg.retriever_mode.upper()
    profile = cfg.dataset_profile
    logger.info("Retriever: %s | Dataset profile: %s", mode, profile)

    if cfg.retriever_mode == "mock":
        return MockRetriever()
    return LiveRetriever()

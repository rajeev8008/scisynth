from __future__ import annotations

import logging

from scisynth.retrieval.contract import RetrievedChunk

logger = logging.getLogger(__name__)


class LiveRetriever:
    """Backed by the indexed store. Filled in Phase 3; returns empty until then."""

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        _ = (query, top_k)
        logger.warning(
            "LiveRetriever: store not wired yet (Phase 3); returning no chunks."
        )
        return []

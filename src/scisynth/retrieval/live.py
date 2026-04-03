from __future__ import annotations

import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from scisynth.config import Settings, get_settings
from scisynth.ingestion.schema import ChunkRecord
from scisynth.retrieval.chunks_io import load_chunks_jsonl
from scisynth.retrieval.contract import RetrievedChunk
from scisynth.retrieval.text import tokenize

logger = logging.getLogger(__name__)


class LiveRetriever:
    """BM25 over ingested chunks.jsonl under ingestion_output_path / dataset_id."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chunks_path = (
            Path(self._settings.ingestion_output_path)
            / self._settings.dataset_id
            / "chunks.jsonl"
        )
        self._chunks: list[ChunkRecord] | None = None
        self._bm25: BM25Okapi | None = None

    def _ensure_loaded(self) -> None:
        if self._chunks is not None:
            return
        path = self._chunks_path
        if not path.is_file():
            logger.warning(
                "LiveRetriever: no chunks at %s — run `scisynth ingest` first.",
                path,
            )
            self._chunks = []
            self._bm25 = None
            return
        self._chunks = load_chunks_jsonl(path)
        if not self._chunks:
            logger.warning("LiveRetriever: %s is empty.", path)
            self._bm25 = None
            return
        tokenized = [tokenize(c.text) for c in self._chunks]
        # BM25Okapi expects at least one document; empty token lists are ok.
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "LiveRetriever: loaded %d chunks from %s",
            len(self._chunks),
            path,
        )

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_loaded()
        if not self._chunks or self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = list(self._bm25.get_scores(q_tokens))
        n = len(scores)
        k = max(0, min(top_k, n))
        if k == 0:
            return []
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        top_scores = [scores[i] for i in order]
        max_s = max(top_scores) if top_scores else 1.0
        out: list[RetrievedChunk] = []
        for i in order:
            c = self._chunks[i]
            raw = scores[i]
            norm = float(raw / max_s) if max_s > 0 else 0.0
            out.append(
                RetrievedChunk(id=c.chunk_id, text=c.text, score=norm),
            )
        return out

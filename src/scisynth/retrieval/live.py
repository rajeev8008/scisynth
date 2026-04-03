from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from scisynth.config import Settings, get_settings
from scisynth.ingestion.schema import ChunkRecord
from scisynth.retrieval.chunks_io import load_chunks_jsonl
from scisynth.retrieval.contract import RetrievedChunk
from scisynth.retrieval.documents_io import load_paper_meta
from scisynth.retrieval.ranking import argsort_descending, minmax_norm, reciprocal_rank_fusion
from scisynth.retrieval.text import tokenize

logger = logging.getLogger(__name__)

_SEMANTIC_WARNED = False


def _semantic_stack_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class LiveRetriever:
    """BM25, optional hybrid semantic + cross-encoder rerank over ingested chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = Path(self._settings.ingestion_output_path) / self._settings.dataset_id
        self._chunks_path = self._root / "chunks.jsonl"
        self._documents_path = self._root / "documents.jsonl"
        self._chunks: list[ChunkRecord] | None = None
        self._bm25: BM25Okapi | None = None
        self._paper_meta: dict[str, str] = {}
        self._embedder = None
        self._embed_matrix: np.ndarray | None = None
        self._cross_encoder = None

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
        self._paper_meta = load_paper_meta(self._documents_path)
        if not self._chunks:
            logger.warning("LiveRetriever: %s is empty.", path)
            self._bm25 = None
            return
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "LiveRetriever: loaded %d chunks from %s",
            len(self._chunks),
            path,
        )

    def _lazy_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            name = self._settings.retrieval_embedding_model
            logger.info("Loading embedding model %s (first call may download weights).", name)
            self._embedder = SentenceTransformer(name)
        return self._embedder

    def _lazy_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            name = self._settings.retrieval_cross_encoder_model
            logger.info("Loading cross-encoder %s (first call may download weights).", name)
            self._cross_encoder = CrossEncoder(name)
        return self._cross_encoder

    def _ensure_embeddings(self) -> None:
        if self._embed_matrix is not None or not self._chunks:
            return
        texts = [c.text for c in self._chunks]
        emb = self._lazy_embedder()
        self._embed_matrix = np.asarray(emb.encode(texts, show_progress_bar=False))

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_loaded()
        if not self._chunks or self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        n = len(self._chunks)
        k_out = max(0, min(top_k, n))
        if k_out == 0:
            return []

        bm25_scores = list(self._bm25.get_scores(q_tokens))
        pipe = self._settings.retrieval_pipeline

        use_hybrid = pipe == "hybrid" and _semantic_stack_available()
        global _SEMANTIC_WARNED
        if pipe == "hybrid" and not _semantic_stack_available() and not _SEMANTIC_WARNED:
            logger.warning(
                "Hybrid retrieval requested but sentence-transformers is not installed; "
                "using BM25 only. Install: pip install -e \".[semantic]\"",
            )
            _SEMANTIC_WARNED = True

        if not use_hybrid:
            return self._pack_bm25_only(bm25_scores, k_out)

        pool = min(n, max(k_out * self._settings.retrieval_candidate_multiplier, k_out))
        bm25_order = argsort_descending(bm25_scores)[:pool]

        self._ensure_embeddings()
        if self._embed_matrix is None or self._embed_matrix.size == 0:
            return self._pack_bm25_only(bm25_scores, k_out)

        q_emb = np.asarray(
            self._lazy_embedder().encode([query], show_progress_bar=False)[0],
        )
        doc_mat = self._embed_matrix
        qn = np.linalg.norm(q_emb)
        dn = np.linalg.norm(doc_mat, axis=1)
        sims = (doc_mat @ q_emb) / (dn * max(qn, 1e-9))
        sem_order = argsort_descending(list(sims))[:pool]

        fused = reciprocal_rank_fusion([bm25_order, sem_order])
        max_pairs = min(
            len(fused),
            self._settings.retrieval_rerank_max_pairs,
            max(k_out * 4, k_out),
        )
        candidate_indices = fused[:max_pairs]

        if self._settings.retrieval_reranker == "cross_encoder" and _semantic_stack_available():
            ce = self._lazy_cross_encoder()
            pairs = [(query, self._chunks[i].text) for i in candidate_indices]
            ce_scores = ce.predict(pairs)
            order = sorted(
                range(len(candidate_indices)),
                key=lambda j: float(ce_scores[j]),
                reverse=True,
            )[:k_out]
            chosen = [candidate_indices[j] for j in order]
            raw_scores = [float(ce_scores[order[i]]) for i in range(len(order))]
        else:
            chosen = candidate_indices[:k_out]
            raw_scores = [float(bm25_scores[i]) for i in chosen]

        norm = minmax_norm(raw_scores)
        return [
            self._to_result(self._chunks[i], score=s)
            for i, s in zip(chosen, norm)
        ]

    def _pack_bm25_only(self, bm25_scores: list[float], k_out: int) -> list[RetrievedChunk]:
        order = argsort_descending(bm25_scores)[:k_out]
        top_scores = [bm25_scores[i] for i in order]
        mx = max(top_scores) if top_scores else 1.0
        out: list[RetrievedChunk] = []
        n = len(order)
        for rank, i in enumerate(order):
            raw = bm25_scores[i]
            if mx > 0:
                norm = float(raw / mx)
            else:
                # All top scores are 0 (rare but valid) — avoid showing 0.00 for every hit.
                norm = 1.0 - (rank / max(n, 1)) * 0.99
            out.append(self._to_result(self._chunks[i], score=norm))
        return out

    def _to_result(self, c: ChunkRecord, *, score: float) -> RetrievedChunk:
        title = self._paper_meta.get(c.paper_id)
        return RetrievedChunk(
            id=c.chunk_id,
            text=c.text,
            score=score,
            paper_id=c.paper_id,
            paper_title=title,
        )

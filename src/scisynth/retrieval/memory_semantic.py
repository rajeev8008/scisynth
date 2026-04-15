from __future__ import annotations

import re
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Optional dependency in lightweight installs
    SentenceTransformer = None

from scisynth.ingestion.schema import ChunkRecord
from scisynth.retrieval.contract import RetrievedChunk
from scisynth.retrieval.ranking import argsort_descending


class InMemorySemanticRetriever:
    """Semantic vector search over an in-memory chunk list for live arXiv papers."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        paper_titles: dict[str, str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._chunks = chunks
        self._paper_titles = paper_titles
        self._use_semantic = SentenceTransformer is not None

        if not chunks:
            self._embed_matrix = None
            return

        if self._use_semantic:
            # Load the lightweight embedding model
            self._embedder = SentenceTransformer(model_name)

            # Embed all chunks into a numpy matrix in RAM
            texts = [c.text for c in chunks]
            self._embed_matrix = np.asarray(self._embedder.encode(texts, show_progress_bar=False))
        else:
            self._embedder = None
            self._embed_matrix = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _lexical_retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Fallback retrieval when sentence-transformers is unavailable."""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)

        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            c_tokens = self._tokenize(chunk.text)
            if not c_tokens:
                continue
            c_set = set(c_tokens)
            overlap = len(q_set & c_set)
            if overlap == 0:
                continue
            # F1-like token overlap score for robust lexical fallback.
            precision = overlap / max(len(c_set), 1)
            recall = overlap / max(len(q_set), 1)
            score = (2 * precision * recall) / max(precision + recall, 1e-9)
            scored.append((score, idx))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievedChunk] = []
        for score, idx in scored[: max(0, min(top_k, len(scored)))]:
            chunk = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    paper_id=chunk.paper_id,
                    paper_title=self._paper_titles.get(chunk.paper_id),
                )
            )
        return out

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        if not self._use_semantic:
            return self._lexical_retrieve(query, top_k=top_k)
        if self._embed_matrix is None:
            return []

        # Embed the search query
        q_emb = np.asarray(self._embedder.encode([query], show_progress_bar=False)[0])

        # Calculate Cosine Similarity
        doc_mat = self._embed_matrix
        qn = np.linalg.norm(q_emb)
        dn = np.linalg.norm(doc_mat, axis=1)
        sims = (doc_mat @ q_emb) / (dn * max(qn, 1e-9))

        n = len(sims)
        k_out = max(0, min(top_k, n))
        if k_out == 0:
            return []

        # Sort by best semantic match
        order = argsort_descending(list(sims))[:k_out]

        out: list[RetrievedChunk] = []
        for i in order:
            c = self._chunks[i]
            title = self._paper_titles.get(c.paper_id)
            score = float(sims[i])

            # Filter out terrible matches
            if score < 0.3:
                continue

            out.append(
                RetrievedChunk(
                    id=c.chunk_id,
                    text=c.text,
                    score=score,
                    paper_id=c.paper_id,
                    paper_title=title,
                )
            )
        return out

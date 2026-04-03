from __future__ import annotations

from rank_bm25 import BM25Okapi

from scisynth.ingestion.schema import ChunkRecord
from scisynth.retrieval.contract import RetrievedChunk
from scisynth.retrieval.ranking import argsort_descending
from scisynth.retrieval.text import tokenize


class InMemoryBM25Retriever:
    """BM25 over an in-memory chunk list (e.g. one on-demand arXiv paper)."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        paper_titles: dict[str, str],
    ) -> None:
        self._chunks = chunks
        self._paper_titles = paper_titles
        if not chunks:
            self._bm25 = None
        else:
            tokenized = [tokenize(c.text) for c in chunks]
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks or self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = list(self._bm25.get_scores(q_tokens))
        n = len(scores)
        k_out = max(0, min(top_k, n))
        if k_out == 0:
            return []
        order = argsort_descending(scores)[:k_out]
        top_scores = [scores[i] for i in order]
        mx = max(top_scores) if top_scores else 1.0
        out: list[RetrievedChunk] = []
        for rank, i in enumerate(order):
            raw = scores[i]
            if mx > 0:
                norm = float(raw / mx)
            else:
                norm = 1.0 - (rank / max(len(order), 1)) * 0.99
            c = self._chunks[i]
            title = self._paper_titles.get(c.paper_id)
            out.append(
                RetrievedChunk(
                    id=c.chunk_id,
                    text=c.text,
                    score=norm,
                    paper_id=c.paper_id,
                    paper_title=title,
                )
            )
        return out

from __future__ import annotations

import math


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = 60,
) -> list[int]:
    """RRF over multiple ranked id lists (e.g. BM25 + semantic)."""
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        for rank, idx in enumerate(ranks, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def argsort_descending(scores: list[float]) -> list[int]:
    """Indices sorted by score descending."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def minmax_norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if math.isclose(hi, lo):
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

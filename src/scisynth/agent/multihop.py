from __future__ import annotations

from scisynth.config import Settings
from scisynth.retrieval.contract import RetrievedChunk


def evidence_insufficient(chunks: list[RetrievedChunk], settings: Settings) -> bool:
    """Return True if first-hop retrieval should trigger a second, refined retrieval."""
    if not chunks:
        return True
    if len(chunks) < settings.rag_evidence_min_chunks:
        return True
    scores = [c.score for c in chunks]
    mx = max(scores)
    mean = sum(scores) / len(scores)
    if mx < settings.rag_evidence_min_max_score:
        return True
    if mean < settings.rag_evidence_min_mean_score:
        return True
    return False


def build_hop2_query(question: str, hop1_chunks: list[RetrievedChunk], *, max_chars: int = 4000) -> str:
    """Smarter second query: question plus top passage text so BM25 can latch on missed terms."""
    parts: list[str] = [question.strip(), "", "Supporting context from first retrieval:"]
    for c in sorted(hop1_chunks, key=lambda x: -x.score)[:5]:
        t = c.text.strip().replace("\n", " ")
        if len(t) > 800:
            t = t[:800] + "…"
        parts.append(f"[{c.id}] {t}")
    merged = "\n".join(parts)
    return merged[:max_chars]


def merge_chunk_lists(
    *lists: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Union by chunk id, keep best score, sort descending by score."""
    best: dict[str, RetrievedChunk] = {}
    for lst in lists:
        for c in lst:
            prev = best.get(c.id)
            if prev is None or c.score > prev.score:
                best[c.id] = c
    return sorted(best.values(), key=lambda x: x.score, reverse=True)

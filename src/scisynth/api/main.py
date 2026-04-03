from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from scisynth.config import get_settings
from scisynth.retrieval import get_retriever


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_retriever()
    yield


app = FastAPI(title="SCISYNTH", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "retriever_mode": s.retriever_mode,
        "dataset_profile": s.dataset_profile,
        "dataset_id": s.dataset_id,
        "ingestion_output_path": s.ingestion_output_path,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=50),
):
    """Run retrieval (mock or live) and return ranked chunks."""
    retriever = get_retriever()
    chunks = retriever.retrieve(q, top_k=top_k)
    return {
        "query": q,
        "top_k": top_k,
        "chunks": [
            {"id": c.id, "text": c.text, "score": c.score} for c in chunks
        ],
    }

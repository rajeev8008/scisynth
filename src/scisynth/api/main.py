from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

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
    }

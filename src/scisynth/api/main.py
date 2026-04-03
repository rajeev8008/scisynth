from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from scisynth.agent import answer_question
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


class AskRequest(BaseModel):
    """Request model for question answering endpoint."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)


class AskResponseCitation(BaseModel):
    """Citation model returned with generated answers."""

    chunk_id: str
    paper_id: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    """Response model for question answering endpoint."""

    question: str
    answer: str
    model: str
    citations: list[AskResponseCitation]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Answer a user question from retrieved context with citations."""
    settings = get_settings()
    try:
        result = answer_question(
            payload.question,
            settings=settings,
            top_k=payload.top_k,
            temperature=payload.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer question.") from exc
    citations = [
        AskResponseCitation(
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            snippet=item.snippet,
            score=item.score,
        )
        for item in result.citations
    ]
    return AskResponse(
        question=result.question,
        answer=result.answer,
        model=result.model,
        citations=citations,
    )

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

from scisynth.agent import (
    answer_question,
    answer_question_with_arxiv,
    answer_question_with_arxiv_discovery,
)
from scisynth.agent.llm_client import generate_answer_text_stream
from scisynth.agent.prompting import build_answer_prompt
from scisynth.agent.service import build_citations, retrieve_chunks_for_answer
from scisynth.config import get_settings
from scisynth.retrieval import get_retriever

logger = logging.getLogger(__name__)


class _SlidingWindowLimiter:
    """Simple per-IP limit: max ``n`` requests per rolling ``window_seconds`` window."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str, max_calls: int) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self._window:
            q.popleft()
        if len(q) >= max_calls:
            return False
        q.append(now)
        return True


_ask_rate_limiter = _SlidingWindowLimiter(window_seconds=60.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """429 when ``RATE_LIMIT_ENABLED`` and too many /ask or /ask/stream calls per IP."""

    async def dispatch(self, request: Request, call_next):
        s = get_settings()
        path = request.url.path
        if (
            s.rate_limit_enabled
            and path in ("/ask", "/ask/stream")
            and request.method == "POST"
        ):
            client = request.client
            ip = client.host if client else "unknown"
            if not _ask_rate_limiter.is_allowed(ip, s.api_rate_limit_per_minute):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded ({s.api_rate_limit_per_minute} requests per minute per IP).",
                    },
                )
        return await call_next(request)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_retriever()
    yield


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to each response for log correlation."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.debug("%s %s [%s]", request.method, request.url.path, rid)
        return response


app = FastAPI(title="SCISYNTH", version="0.1.0", lifespan=lifespan)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIdMiddleware)


@app.get("/health")
def health(
    deep: bool = Query(
        False,
        description="If true, probe LLM base URL (requires API key).",
    ),
):
    """Liveness plus optional dependency checks."""
    s = get_settings()
    chunks_path = Path(s.ingestion_output_path) / s.dataset_id / "chunks.jsonl"
    key = (s.llm_api_key or s.openai_api_key).strip()
    checks: dict[str, object] = {
        "chunks_file": {
            "exists": chunks_path.is_file(),
            "path": str(chunks_path.resolve()),
        },
        "llm_api_key": {"configured": bool(key)},
        "retrieval_pipeline": s.retrieval_pipeline,
    }
    status = "ok"
    if s.retriever_mode == "live" and not chunks_path.is_file():
        status = "degraded"

    llm_probe: dict[str, object] | None = None
    if deep and key:
        llm_probe = _probe_llm_endpoint(s, key)
        if llm_probe.get("ok") is False:
            status = "degraded"

    return {
        "status": status,
        "retriever_mode": s.retriever_mode,
        "dataset_profile": s.dataset_profile,
        "dataset_id": s.dataset_id,
        "ingestion_output_path": s.ingestion_output_path,
        "checks": checks,
        "llm_probe": llm_probe,
    }


def _probe_llm_endpoint(s, api_key: str) -> dict[str, object]:
    """Lightweight GET toward /models on OpenAI-compatible hosts."""
    base = s.llm_base_url.rstrip("/")
    probe_url = base + "/models"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                probe_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "url": probe_url,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": probe_url}


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
            {
                "id": c.id,
                "text": c.text,
                "score": c.score,
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
            }
            for c in chunks
        ],
    }


class AskRequest(BaseModel):
    """Request model for question answering endpoint."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    arxiv_url_or_id: str | None = Field(
        default=None,
        description="Optional. If set (and discovery is off), fetch this arXiv paper and answer from it only.",
    )
    arxiv_discovery: bool = Field(
        default=False,
        description="If true, run arXiv keyword search for ``question`` and answer over top results.",
    )

    @model_validator(mode="after")
    def _one_arxiv_mode(self):
        ref = (self.arxiv_url_or_id or "").strip()
        if self.arxiv_discovery and ref:
            raise ValueError("Use either arxiv_discovery or arxiv_url_or_id, not both.")
        return self


class AskResponseCitation(BaseModel):
    """Citation model returned with generated answers."""

    chunk_id: str
    paper_id: str
    paper_title: str | None = None
    snippet: str
    score: float


class AskResponse(BaseModel):
    """Response model for question answering endpoint."""

    question: str
    answer: str
    model: str
    citations: list[AskResponseCitation]
    retrieval_hops_used: int = Field(
        default=1,
        description="1 = single retrieval; 2 = second hop ran (multi-hop RAG).",
    )


def _dispatch_ask(payload: AskRequest):
    """Run the appropriate answer path; raises ValueError / RuntimeError for HTTP mapping."""
    settings = get_settings()
    if payload.arxiv_discovery:
        return answer_question_with_arxiv_discovery(
            payload.question,
            settings=settings,
            top_k=payload.top_k,
            temperature=payload.temperature,
        )
    ref = (payload.arxiv_url_or_id or "").strip()
    if ref:
        try:
            return answer_question_with_arxiv(
                payload.question,
                ref,
                settings=settings,
                top_k=payload.top_k,
                temperature=payload.temperature,
            )
        except ValueError:
            raise
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"arXiv or network error while fetching the paper: {exc!s}",
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Could not load that arXiv paper. Check the id, try again later, or use indexed mode.",
            ) from exc
    return answer_question(
        payload.question,
        settings=settings,
        top_k=payload.top_k,
        temperature=payload.temperature,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Answer a user question from retrieved context with citations."""
    try:
        result = _dispatch_ask(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "arxiv" in msg or "network" in msg:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail="Failed to answer question.") from exc
    citations = [
        AskResponseCitation(
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            paper_title=item.paper_title,
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
        retrieval_hops_used=result.retrieval_hops_used,
    )


def _sse_line(obj: dict[str, object]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(payload: AskRequest) -> StreamingResponse:
    """Stream answer tokens (SSE). Final line includes citations metadata when available."""

    def gen():
        settings = get_settings()
        try:
            if payload.arxiv_discovery:
                result = answer_question_with_arxiv_discovery(
                    payload.question,
                    settings=settings,
                    top_k=payload.top_k,
                    temperature=payload.temperature,
                )
                yield _sse_line({"type": "meta", "retrieval_hops_used": result.retrieval_hops_used})
                yield _sse_line({"type": "token", "text": result.answer})
                yield _sse_line(
                    {
                        "type": "done",
                        "citations": [
                            {
                                "chunk_id": c.chunk_id,
                                "paper_id": c.paper_id,
                                "paper_title": c.paper_title,
                                "snippet": c.snippet,
                                "score": c.score,
                            }
                            for c in result.citations
                        ],
                        "model": result.model,
                    },
                )
                return
            ref = (payload.arxiv_url_or_id or "").strip()
            if ref:
                result = answer_question_with_arxiv(
                    payload.question,
                    ref,
                    settings=settings,
                    top_k=payload.top_k,
                    temperature=payload.temperature,
                )
                yield _sse_line({"type": "meta", "retrieval_hops_used": result.retrieval_hops_used})
                yield _sse_line({"type": "token", "text": result.answer})
                yield _sse_line(
                    {
                        "type": "done",
                        "citations": [
                            {
                                "chunk_id": c.chunk_id,
                                "paper_id": c.paper_id,
                                "paper_title": c.paper_title,
                                "snippet": c.snippet,
                                "score": c.score,
                            }
                            for c in result.citations
                        ],
                        "model": result.model,
                    },
                )
                return

            chunks_final, hops_used = retrieve_chunks_for_answer(
                payload.question,
                settings=settings,
                retriever=None,
                top_k=payload.top_k,
            )
            yield _sse_line({"type": "meta", "retrieval_hops_used": hops_used})
            if not chunks_final:
                yield _sse_line(
                    {
                        "type": "error",
                        "detail": "No indexed context matched this question.",
                    },
                )
                yield _sse_line({"type": "done", "citations": [], "model": settings.llm_model})
                return

            prompt = build_answer_prompt(
                payload.question,
                chunks_final,
                retrieval_hops_used=hops_used,
            )
            acc: list[str] = []
            for piece in generate_answer_text_stream(
                settings,
                prompt,
                temperature=payload.temperature
                if payload.temperature is not None
                else settings.answer_temperature,
                max_output_tokens=settings.llm_max_output_tokens,
            ):
                acc.append(piece)
                yield _sse_line({"type": "token", "text": piece})
            full = "".join(acc)
            cites = build_citations(chunks_final)
            yield _sse_line(
                {
                    "type": "done",
                    "citations": [
                        {
                            "chunk_id": c.chunk_id,
                            "paper_id": c.paper_id,
                            "paper_title": c.paper_title,
                            "snippet": c.snippet,
                            "score": c.score,
                        }
                        for c in cites
                    ],
                    "model": settings.llm_model,
                    "answer": full,
                },
            )
        except ValueError as exc:
            yield _sse_line({"type": "error", "detail": str(exc)})
        except RuntimeError as exc:
            yield _sse_line({"type": "error", "detail": str(exc)})
        except Exception as exc:
            logger.exception("ask_stream failed")
            yield _sse_line({"type": "error", "detail": "Streaming failed."})

    return StreamingResponse(gen(), media_type="text/event-stream")


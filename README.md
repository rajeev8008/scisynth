# SCISYNTH

Science-synthesis style RAG project. **Phase 1** provides the repo layout, retrieval contract, mock/live toggle, and a health API.

## Setup

Python 3.11+ recommended.

```bash
cd SCISYNTH
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

(`[dev]` adds `httpx` for FastAPI `TestClient` in tests; use `pip install -e .` if you only run the server.)

Copy `.env.example` to `.env` and adjust if needed.

## Run

```bash
scisynth
```

Then open `http://127.0.0.1:8000/health`.

- `RETRIEVER_MODE=mock` — deterministic fixture chunks (default).
- `RETRIEVER_MODE=live` — wired in Phase 3 (currently returns no chunks with a log warning).

## Layout

| Area | Role |
|------|------|
| `src/scisynth/retrieval/` | `Retriever` contract, `get_retriever()`, mock/live |
| `src/scisynth/ingestion/` | Corpus → index (Phase 2) |
| `src/scisynth/agent/` | Orchestration / LLM (Phase 4) |
| `src/scisynth/api/` | HTTP API |

See `.cursorrules` for AI/editor conventions.

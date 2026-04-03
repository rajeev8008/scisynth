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

## Ingest (Phase 2)

Run one command to ingest the configured corpus profile:

```bash
scisynth ingest
```

Install Hugging Face loading when needed:

```bash
pip install -e ".[dev]"
```

Config is env-driven via `src/scisynth/config.py`:

- `DATASET_SOURCE=local|arxiv|huggingface`
- `DATASET_PROFILE=fixture|full`
- `DATASET_ID` (version label, e.g. `fixture-v1`)
- `DATASET_FULL_PATH` (source path when profile is `full`)
- `INGESTION_OUTPUT_PATH` (default `data/processed`)
- `INGESTION_RAW_PATH` (optional raw mirrors, e.g. arXiv snapshot)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `ARXIV_QUERY`, `ARXIV_MAX_RESULTS`, `ARXIV_TOPIC`, `ARXIV_PERSIST_RAW`
- `HF_PRESET=qasper|scifact_corpus`, `HF_SPLIT`, `HF_MAX_ROWS`, `HF_QASPER_REVISION`, `HF_SCIFACT_PARQUET_GLOB`

Ingestion writes:

- `documents.jsonl`
- `chunks.jsonl`
- `manifest.json`

under `INGESTION_OUTPUT_PATH/DATASET_ID/`.

Example arXiv ingest:

```bash
set DATASET_SOURCE=arxiv
set DATASET_ID=arxiv-cscl-v1
set ARXIV_QUERY=cat:cs.CL
set ARXIV_MAX_RESULTS=10
scisynth ingest
```

Example Hugging Face slice (QASPER via parquet revision; requires `datasets`):

```bash
set DATASET_SOURCE=huggingface
set HF_PRESET=qasper
set HF_SPLIT=train
set HF_MAX_ROWS=50
set DATASET_ID=qasper-train-50
scisynth ingest
```

CLI overrides (optional):

```bash
scisynth ingest --source huggingface --hf-preset scifact_corpus --hf-max-rows 100 --dataset-id scifact-corpus-100
```

## Layout

| Area | Role |
|------|------|
| `src/scisynth/retrieval/` | `Retriever` contract, `get_retriever()`, mock/live |
| `src/scisynth/ingestion/` | Corpus → index (Phase 2) |
| `src/scisynth/agent/` | Orchestration / LLM (Phase 4) |
| `src/scisynth/api/` | HTTP API |

See `.cursorrules` for AI/editor conventions.

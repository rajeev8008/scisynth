# SCISYNTH

**SciSynth** is a research-style Q&A stack: you ask a question, it **retrieves** evidence (from your ingested index or an on-demand arXiv paper), and the **LLM** answers with **citations** to chunk ids and paper titles.

**Multi-hop RAG (default):** the first retrieval pass may not be enough. If evidence looks weak (too few chunks or low scores), the agent runs a **second retrieval** with an **expanded query** built from the top passages of the first hop, then **merges and deduplicates** chunks before answering. Configure with `RAG_*` settings in `.env` (see `config.py`).

## Setup

Python 3.11+ recommended.

```bash
cd SCISYNTH
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

(`[dev]` adds test deps; `pip install -e ".[semantic]"` enables hybrid retrieval; `pip install -e ".[ui]"` adds the Gradio demo.)

Copy `.env.example` to **`.env`** in the **project root** (next to `pyproject.toml`) and add your keys. Settings load from that path **even if you start the app from another folder** — your previous “missing API key” errors were often from the shell’s working directory, not from a missing file.

## Run

```bash
scisynth
```

Then open `http://127.0.0.1:8000/health`.

- `RETRIEVER_MODE=mock` — deterministic fixture chunks (default).
- `RETRIEVER_MODE=live` — reads `INGESTION_OUTPUT_PATH/DATASET_ID/chunks.jsonl` (run `scisynth ingest` first).
- `RETRIEVAL_PIPELINE=hybrid` (default: BM25 + embeddings + cross-encoder rerank when `pip install -e ".[semantic]"` is installed; otherwise BM25-only with a warning) or `bm25` for lexical-only.

Try search (live or mock): `http://127.0.0.1:8000/search?q=neural&top_k=5`.

Health: `GET /health` (add `?deep=1` to probe the LLM `/models` endpoint when an API key is set). Responses include `X-Request-ID` for debugging.

## Ask (Phase 4)

Question answering now runs as: question -> retrieve -> LLM -> cited answer.

Endpoint:

```bash
POST /ask
{
  "question": "What improves synthesis in long documents?",
  "top_k": 5,
  "temperature": 0.2
}
```

Optional fields:

- `arxiv_url_or_id` — fetch **one** paper (URL or id). With `ARXIV_FETCH_FULL_PDF=true` (default), the service downloads the PDF and extracts text via **PyMuPDF** (falls back to title+abstract if the PDF fails).
- `arxiv_discovery` — if `true`, run an arXiv **keyword search** using `question` and answer over the top `ARXIV_DISCOVERY_MAX_RESULTS` hits (also supports full PDF per paper when `ARXIV_DISCOVERY_USE_FULL_PDF=true`). Do not combine with `arxiv_url_or_id`.

**Streaming:** `POST /ask/stream` with the same JSON body returns **SSE** (`text/event-stream`): events `meta` (retrieval hops), `token` (text deltas for indexed mode), `done` (citations + model), or `error`.

**Rate limits:** set `RATE_LIMIT_ENABLED=true` to enforce `API_RATE_LIMIT_PER_MINUTE` per client IP on `/ask` and `/ask/stream` (sliding window).

The Gradio UI offers **Ingested index**, **Single arXiv paper**, or **Search arXiv (discovery)**, optional **streaming** for indexed mode, and shows **retrieval hops** plus **sources** explicitly.

Response includes:

- `answer`
- `citations` (chunk_id, paper_id, paper_title, snippet, score)
- `model`
- `retrieval_hops_used` (1 or 2 when multi-hop RAG runs a second pass)

Required env for generation:

- `LLM_BASE_URL`
- `LLM_API_KEY` (or `OPENAI_API_KEY`)
- `LLM_MODEL`
- `LLM_MAX_OUTPUT_TOKENS`

If retrieval returns no chunks, the system returns: "I do not know based on the available indexed context."

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
- `ARXIV_QUERY`, `ARXIV_MAX_RESULTS`, `ARXIV_TOPIC`, `ARXIV_PERSIST_RAW`, `ARXIV_FETCH_FULL_PDF`, `ARXIV_PDF_MAX_BYTES`, `ARXIV_PDF_MAX_EXTRACT_CHARS`
- Discovery API/UI: `ARXIV_DISCOVERY_MAX_RESULTS`, `ARXIV_DISCOVERY_TOPIC`, `ARXIV_DISCOVERY_USE_FULL_PDF`
- `HF_PRESET=qasper|scifact_corpus`, `HF_SPLIT`, `HF_MAX_ROWS`, `HF_QASPER_REVISION`, `HF_SCIFACT_PARQUET_GLOB`

Ingestion writes:

- `documents.jsonl`
- `chunks.jsonl`
- `manifest.json`

under `INGESTION_OUTPUT_PATH/DATASET_ID/`.

## Retrieve (Phase 3)

1. Ingest so `chunks.jsonl` exists for your `DATASET_ID` (see above).
2. Set `RETRIEVER_MODE=live` and match `INGESTION_OUTPUT_PATH` and `DATASET_ID` to that run.
3. Use `GET /search?q=...` or call `get_retriever().retrieve(...)` in code.

- **bm25:** lexical BM25 only (`rank-bm25`).
- **hybrid:** BM25 + sentence embeddings + optional **cross-encoder** rerank (`sentence-transformers`; first run downloads models).

## Eval (Phase 4 lightweight)

Run frozen end-to-end eval (uses full answer pipeline, not mock outputs):

```bash
scisynth eval
```

Inputs:

- `EVAL_QUESTIONS_PATH` JSONL with `id`, `question`, and optional `rubric_keywords` (list of strings for a cheap coverage score)

Outputs:

- timestamped CSV under `EVAL_RESULTS_DIR` (default `eval/results`): `keyword_overlap`, `retrieval_hops_used`, `has_chunk_citation_markers`, `rubric_keyword_coverage`, etc.

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

## Web UI (Gradio)

```bash
pip install -e ".[ui]"
scisynth-ui
```

Opens a browser UI (default `http://127.0.0.1:7860`) that calls the same answering pipeline as `/ask` (plus arXiv discovery and optional streaming).

**Local PDFs:** place `.pdf` files under your corpus folder (e.g. `DATASET_FULL_PATH` with `DATASET_PROFILE=full`); ingestion extracts text with PyMuPDF alongside `.md` / `.txt`.

## CI

GitHub Actions runs `pytest` on push/PR (`.github/workflows/ci.yml`).

See `.cursorrules` for AI/editor conventions.

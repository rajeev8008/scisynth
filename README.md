# SCISYNTH: Multi-Agent Deep Research System

**SciSynth** is an advanced AI research assistant powered by LangGraph, LangChain, and RAG. It transforms short questions or broad topics into fully researched, well-cited academic reports.

It operates in two primary modes:
1. **Quick Q&A (Single-Agent):** You ask a question, it retrieves evidence from an ingested corpus or arXiv, and an LLM generates a cited answer.
2. **Deep Research (Multi-Agent):** You provide a topic. A team of collaborative AI agents (Planner, Researcher, Writer, Reviewer, Synthesizer) plan an outline, fetch live papers from arXiv (or use ingested data), draft sections, review for quality, and synthesize a cohesive final report.

##  Features

- **Multi-Agent Orchestration (LangGraph):** Cyclic workflows with feedback loops ensuring high-quality, grounded research.
- **Live arXiv Discovery:** The Deep Research mode can fetch real papers from arXiv on-the-fly. No pre-ingestion required!
- **Multi-Hop RAG:** Automatically performs secondary, smarter retrievals if initial evidence is weak.
- **Hybrid Retrieval:** Combines BM25 lexical search with semantic embeddings and cross-encoder reranking.
- **Modern Chat UI:** Built with Chainlit, offering real-time progress tracking for background agent tasks.
- **Strict Citation Tracking:** Every claim is backed by a specific chunk ID and paper title.

##  Setup

Python 3.11+ recommended.

```bash
git clone https://github.com/rajeev8008/scisynth.git
cd scisynth
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate # On Unix/macOS

# Install with research and UI dependencies
pip install -e ".[research,ui,semantic,dev]"
```

### Configuration

Copy `.env.example` to **`.env`** in the project root and add your API keys. Fast generation recommends the Groq API (e.g., `llama-3.1-8b-instant`), but any OpenAI-compatible endpoint works.

Required `.env` variables for agent functionality:
- `LLM_BASE_URL` (e.g., `https://api.groq.com/openai/v1`)
- `LLM_API_KEY`
- `LLM_MODEL` (e.g., `llama-3.1-8b-instant`)

##  Running the UI

Start the Chainlit web interface:

```bash
scisynth-ui
```
Open `http://localhost:7860` in your browser.

### Using the UI

Type directly in the chat box:
- **`What improves retrieval quality in long-document QA?`**: Triggers Quick Q&A against the ingested index.
- **`/research How do transformer models handle long-context scientific QA?`**: Triggers the Deep Research Multi-Agent pipeline, fetching live papers from arXiv.
- **`/research-index <topic>`**: Runs Deep Research using *only* your pre-ingested local index.
- **`/arxiv 2005.11401 How does RAG work?`**: Fetches the specified arXiv paper, reads it, and answers your question.
- **`/discover What are recent advances in RAG?`**: Searches arXiv, summarizes the top 5 papers, and answers.

##  System Architecture

The core innovation is the **Deep Research LangGraph Pipeline**:

1. **Planner Node:** Decomposes the topic into 3-5 sub-sections.
2. **Researcher Node:** For each section, queries arXiv (or local index), downloads PDFs, chunks, and retrieves evidence.
3. **Writer Node:** Synthesizes a draft for the section, citing the retrieved evidence.
4. **Reviewer Node:** Critiques the draft. If citations are missing or it hallucinates, it routes back to the Researcher (for more data) or Writer (for a rewrite).
5. **Synthesizer Node:** Stitches accepted sections into a final markdown report.

##  Ingestion Pipeline (Optional)

If you want to query your own local documents or a specific HuggingFace dataset, you can build a local index.

```bash
# Example: Ingest HuggingFace dataset
# Edit .env to set DATASET_SOURCE=huggingface, HF_PRESET=qasper
scisynth ingest
```

Data is written to `data/processed/<DATASET_ID>/` as `chunks.jsonl` and `documents.jsonl`.

##  Evaluation
Run the frozen benchmarking suite:

```bash
scisynth eval
```
Outputs a CSV report in `eval/results/` tracking keyword coverage, retrieval hops, and citation formatting.

##  Layout

| Area | Role |
|------|------|
| `src/scisynth/research/` | LangGraph multi-agent orchestration, state logic, specialized nodes |
| `src/scisynth/agent/` | Single-agent Q&A and multi-hop routing logic |
| `src/scisynth/retrieval/` | BM25, embeddings, cross-encoder ranking, memory retriever |
| `src/scisynth/ingestion/` | PDF extraction, chunking, downloading from arXiv/HF |
| `src/scisynth/ui/` | Chainlit web interface |

##  License

MIT License

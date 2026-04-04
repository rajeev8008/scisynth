from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "ingestion" / "fixtures" / "v1"
).as_posix()
# Always load secrets from the repo root (next to pyproject.toml), not from the process cwd.
# Otherwise `scisynth-ui` started from another folder never sees `.env`.
_DOTENV_PATH = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_DOTENV_PATH) if _DOTENV_PATH.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    retriever_mode: Literal["mock", "live"] = Field(
        default="mock",
        description="mock: deterministic chunks; live: backed by store (Phase 3+).",
    )
    dataset_source: Literal["local", "arxiv", "huggingface"] = Field(
        default="local",
        description="local: filesystem; arxiv: arXiv API; huggingface: HF dataset slice.",
    )
    dataset_profile: Literal["fixture", "full"] = Field(
        default="fixture",
        description="fixture: tiny sample; full: production corpus (Phase 2+).",
    )
    dataset_id: str = Field(
        default="fixture-v1",
        description="Versioned dataset id for ingestion outputs and retrieval selection.",
    )
    dataset_fixture_path: str = Field(
        default=_DEFAULT_FIXTURE_PATH,
        description="Path to versioned fixture corpus directory.",
    )
    dataset_full_path: str = Field(
        default="data/raw/full",
        description="Path to full corpus directory.",
    )
    ingestion_output_path: str = Field(
        default="data/processed",
        description="Root path for ingestion outputs.",
    )
    ingestion_raw_path: str = Field(
        default="data/raw",
        description="Root path for optional raw snapshots (e.g. arXiv mirrors).",
    )
    chunk_size: int = Field(
        default=800,
        description="Chunk size in characters for ingestion.",
    )
    chunk_overlap: int = Field(
        default=120,
        description="Chunk overlap in characters for ingestion.",
    )
    arxiv_query: str = Field(
        default="cat:cs.CL",
        description="arXiv search query used when dataset_source=arxiv.",
    )
    arxiv_max_results: int = Field(
        default=5,
        description="Maximum number of arXiv papers to ingest.",
    )
    arxiv_topic: str = Field(
        default="arxiv",
        description="Topic label assigned to arXiv-loaded documents.",
    )
    arxiv_persist_raw: bool = Field(
        default=False,
        description="When true, write arXiv fetched bodies under ingestion_raw_path/arxiv/.",
    )
    arxiv_fetch_full_pdf: bool = Field(
        default=True,
        description="When true, download and extract PDF text for on-demand arXiv fetches (single paper + discovery).",
    )
    arxiv_pdf_max_bytes: int = Field(
        default=25_000_000,
        ge=1_000_000,
        description="Max PDF download size (bytes).",
    )
    arxiv_pdf_max_extract_chars: int = Field(
        default=1_500_000,
        ge=10_000,
        description="Max characters extracted from a PDF into text.",
    )
    arxiv_discovery_max_results: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Max papers when using arXiv keyword discovery for a question.",
    )
    arxiv_discovery_topic: str = Field(
        default="arxiv-discovery",
        description="Topic label for discovery-ingested documents.",
    )
    arxiv_discovery_use_full_pdf: bool = Field(
        default=True,
        description="Extract full PDF text for each discovery result (slower, richer).",
    )
    hf_preset: Literal["qasper", "scifact_corpus"] = Field(
        default="qasper",
        description="Which HF preset to load when dataset_source=huggingface.",
    )
    hf_split: str = Field(
        default="train",
        description="Split name for QASPER (ignored for SciFact corpus parquet).",
    )
    hf_max_rows: int = Field(
        default=20,
        description="Max rows to load from HF (0 = full split / corpus train).",
    )
    hf_qasper_revision: str = Field(
        default="refs/convert/parquet",
        description="Hub revision for QASPER parquet (avoids deprecated script loader).",
    )
    hf_scifact_parquet_glob: str = Field(
        default="hf://datasets/allenai/scifact@refs/convert/parquet/corpus/train/*.parquet",
        description="Parquet glob for SciFact corpus shards on the Hub.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base URL for chat completions.",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the configured LLM endpoint.",
    )
    openai_api_key: str = Field(
        default="",
        description="Compatibility key; used when llm_api_key is empty.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for answer generation.",
    )
    llm_max_output_tokens: int = Field(
        default=500,
        description="Max tokens generated by the answering model.",
    )
    answer_top_k: int = Field(
        default=5,
        description="Default retrieval top-k for answer generation.",
    )
    answer_temperature: float = Field(
        default=0.2,
        description="Default sampling temperature for answer generation.",
    )
    rag_multi_hop: bool = Field(
        default=True,
        description="If true, run a second retrieval when first-hop evidence looks weak (multi-hop RAG).",
    )
    rag_max_hops: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Maximum retrieval hops (2 = first pass + refined second pass).",
    )
    rag_hop1_top_k: int | None = Field(
        default=None,
        description="First-hop top-k; None uses answer_top_k.",
    )
    rag_hop2_top_k: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Second-hop top-k (often larger to capture missed evidence).",
    )
    rag_evidence_min_chunks: int = Field(
        default=2,
        ge=1,
        description="Below this chunk count, trigger hop 2 (if multi-hop enabled).",
    )
    rag_evidence_min_max_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="If best chunk score is below this, trigger hop 2.",
    )
    rag_evidence_min_mean_score: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="If mean chunk score is below this, trigger hop 2.",
    )
    research_max_sections: int = Field(
        default=5,
        ge=2,
        le=8,
        description="Maximum sections the planner can create for deep research.",
    )
    research_max_review_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max reviewer loop iterations per section before force-accept.",
    )
    eval_questions_path: str = Field(
        default="eval/questions/frozen_questions.jsonl",
        description="Path to frozen evaluation question set.",
    )
    eval_results_dir: str = Field(
        default="eval/results",
        description="Directory where evaluation CSV outputs are written.",
    )
    retrieval_pipeline: Literal["bm25", "hybrid"] = Field(
        default="hybrid",
        description="bm25: lexical only; hybrid: BM25 + embeddings + optional cross-encoder rerank.",
    )
    retrieval_candidate_multiplier: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Pool size factor vs top_k for hybrid fusion.",
    )
    retrieval_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model id for semantic leg of hybrid retrieval.",
    )
    retrieval_cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="CrossEncoder model for second-stage reranking.",
    )
    retrieval_reranker: Literal["none", "cross_encoder"] = Field(
        default="cross_encoder",
        description="Second-stage reranker; none uses RRF ordering only.",
    )
    retrieval_rerank_max_pairs: int = Field(
        default=32,
        ge=4,
        le=256,
        description="Max candidates passed to the cross-encoder.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Retries for transient LLM HTTP failures.",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Read timeout for LLM HTTP calls (seconds).",
    )
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_port: int = Field(
        default=7860,
        description="Port for the Gradio demo when launched via scisynth-ui.",
    )
    rate_limit_enabled: bool = Field(
        default=False,
        description="When true, apply rate limits to /ask and /ask/stream (enable in production).",
    )
    api_rate_limit_per_minute: int = Field(
        default=90,
        ge=1,
        le=10_000,
        description="Max /ask requests per client IP per minute.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Clear cached settings (e.g. after editing `.env` in the same process)."""
    get_settings.cache_clear()
    return get_settings()

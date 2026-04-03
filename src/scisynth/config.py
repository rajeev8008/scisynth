from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "ingestion" / "fixtures" / "v1"
).as_posix()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()

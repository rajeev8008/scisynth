from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    dataset_profile: str = Field(
        default="fixture",
        description="fixture: tiny sample; full: production corpus (Phase 2+).",
    )
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()

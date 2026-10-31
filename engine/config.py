from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_in_memory_bytes: int = 50 * 1024 * 1024
    max_total_bytes: int = 500 * 1024 * 1024
    sample_rows_for_charts: int = 200_000
    correlation_threshold: float = 0.3
    cramers_v_threshold: float = 0.3
    max_corr_pairs: int = 5
    max_charts_per_section: int = 12
    retention_hours: int = 24
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    allowed_extensions: tuple[str, ...] = (".csv", ".tsv", ".xlsx", ".xls", ".json")

    class Config:
        env_file = ".env"


settings = Settings()

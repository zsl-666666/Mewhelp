from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    context_token_budget: int = 3000
    structured_output_method: Literal[
        "function_calling", "json_schema"
    ] = "function_calling"
    company_name: str = "好物商城"
    service_hours: str = "9:00-21:00"
    return_window_days: int = 7


@lru_cache
def get_settings() -> Settings:
    return Settings()

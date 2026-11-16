from fastapi import Depends
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )


def get_chat_model(settings: Settings = Depends(get_settings)) -> ChatOpenAI:
    return build_chat_model(settings)

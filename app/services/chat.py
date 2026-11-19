import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import Settings
from app.prompts.customer_service import build_customer_service_prompt
from app.services.context import trim_history
from app.sse import format_sse

logger = logging.getLogger(__name__)


def extract_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


async def stream_chat(
    history: list[BaseMessage],
    model: BaseChatModel,
    settings: Settings,
) -> AsyncIterator[str]:
    try:
        trimmed = trim_history(history, settings.context_token_budget)
        system_message = build_customer_service_prompt().format_messages(
            company_name=settings.company_name,
            service_hours=settings.service_hours,
            return_window_days=settings.return_window_days,
        )[0]
        messages = [system_message, *trimmed.messages]

        async for chunk in model.astream(messages):
            text = extract_text(chunk)
            if text:
                yield format_sse({"delta": text})
        yield format_sse({"done": True, "trimmed": trimmed.trimmed})
    except Exception:
        logger.exception("chat stream failed")
        yield format_sse(
            {
                "code": "upstream_error",
                "message": "模型服务暂时不可用，请稍后重试",
            },
            event="error",
        )

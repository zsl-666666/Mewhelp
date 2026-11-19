"""单轮工具调用编排：加载历史、调用模型、执行工具并落库。"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import Settings
from app.prompts.customer_service import (
    TOOL_USAGE_RULES,
    build_customer_service_prompt,
)
from app.repositories.conversations import append_message, load_history
from app.services.context import repair_tool_pairs, trim_history
from app.sse import format_sse
from app.tools.registry import (
    ToolExecutionEvent,
    ToolExecutionResult,
    ToolRegistry,
)

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


def build_tool_message_content(
    tool_call: dict[str, Any],
    result: ToolExecutionResult,
    max_chars: int,
) -> str:
    """生成 tool 消息的 JSON envelope，``result`` 按 ``max_chars`` 截断。"""
    content = result.content or ""
    original_chars = len(content)
    payload: dict[str, Any] = {
        "tool": str(tool_call.get("name") or ""),
        "arguments": dict(tool_call.get("args") or {}),
        "ok": result.ok,
        "result": content[:max_chars],
        "truncated": original_chars > max_chars,
        "original_chars": original_chars,
        "attempts": result.attempts,
        "duration_ms": result.duration_ms,
    }
    if not result.ok:
        payload["error_code"] = result.error_code
        payload["retryable"] = result.retryable
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tool_event_frame(event: ToolExecutionEvent) -> str:
    """把 registry 事件映射为 spec 8.1 的 ``tool`` SSE 帧。"""
    base = {
        "tool_call_id": event.tool_call_id,
        "name": event.name,
        "label": event.label,
    }
    if event.status == "started":
        data = {**base, "status": "running", "arguments": event.arguments}
    elif event.status == "retrying":
        data = {
            **base,
            "status": "retrying",
            "attempt": event.attempt,
            "max_attempts": event.max_attempts,
            "error_code": event.error_code,
            "error": event.error,
        }
    elif event.status == "completed":
        result = event.result
        data = {
            **base,
            "status": "completed",
            "ok": True,
            "attempts": result.attempts if result is not None else event.attempt,
            "duration_ms": result.duration_ms if result is not None else 0,
        }
    else:  # timeout / failed
        result = event.result
        data = {
            **base,
            "status": event.status,
            "ok": False,
            "error_code": event.error_code,
            "error": event.error,
            "retryable": result.retryable if result is not None else False,
            "attempts": result.attempts if result is not None else event.attempt,
            "duration_ms": result.duration_ms if result is not None else 0,
        }
    return format_sse(data, event="tool")


async def stream_chat(
    *,
    conversation_id: int,
    user_id: str,
    session_factory,
    registry: ToolRegistry,
    model: BaseChatModel,
    settings: Settings,
) -> AsyncIterator[str]:
    pending_text = ""

    def load_messages() -> list[BaseMessage]:
        with session_factory() as session:
            return load_history(session, conversation_id)

    def persist_assistant(
        text: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        with session_factory() as session:
            append_message(
                session,
                conversation_id,
                role="assistant",
                content=text or None,
                tool_calls=tool_calls,
            )

    def persist_tool(content: str, tool_call_id: str) -> None:
        with session_factory() as session:
            append_message(
                session,
                conversation_id,
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )

    try:
        history = await asyncio.to_thread(load_messages)
        trimmed_history = trim_history(history, settings.context_token_budget)
        trimmed = trimmed_history.trimmed
        messages = repair_tool_pairs(trimmed_history.messages)

        base_system = build_customer_service_prompt().format_messages(
            company_name=settings.company_name,
            service_hours=settings.service_hours,
            return_window_days=settings.return_window_days,
        )[0]
        system_message = SystemMessage(
            content=f"{base_system.content}\n\n{TOOL_USAGE_RULES}"
        )
        messages = [system_message, *messages]

        yield format_sse(
            {"conversation_id": conversation_id, "user_id": user_id},
            event="meta",
        )

        tools = registry.bindable_tools()
        bound_model = model.bind_tools(tools) if tools else model

        accumulated: AIMessageChunk | None = None
        async for chunk in bound_model.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = extract_text(chunk)
            if text:
                pending_text += text
                yield format_sse({"delta": text})

        tool_calls = list(accumulated.tool_calls) if accumulated is not None else []
        assistant_text = extract_text(accumulated) if accumulated is not None else ""

        if not tool_calls:
            if assistant_text:
                await asyncio.to_thread(persist_assistant, assistant_text)
                pending_text = ""
            yield format_sse(
                {
                    "done": True,
                    "conversation_id": conversation_id,
                    "trimmed": trimmed,
                    "tool_calls": [],
                }
            )
            return

        tool_call = dict(tool_calls[0])
        if len(tool_calls) > 1:
            logger.warning(
                "conversation %s: model returned %s tool calls, executing only %s",
                conversation_id,
                len(tool_calls),
                tool_call.get("name"),
            )
        tool_call_id = str(tool_call.get("id") or "")

        await asyncio.to_thread(
            persist_assistant, assistant_text, [tool_call]
        )
        pending_text = ""
        messages.append(
            AIMessage(content=assistant_text, tool_calls=[tool_call])
        )

        result: ToolExecutionResult | None = None
        async for event in registry.execute_stream(
            tool_call,
            injected_args={"conversation_id": conversation_id},
            timeout_seconds=settings.tool_timeout_seconds,
            max_retries=settings.tool_max_retries,
            retry_delay_seconds=settings.tool_retry_delay_seconds,
        ):
            yield _tool_event_frame(event)
            if event.result is not None:
                result = event.result
        if result is None:
            result = ToolExecutionResult(
                ok=False,
                content=json.dumps(
                    {"ok": False, "error": "工具执行失败"}, ensure_ascii=False
                ),
                error="工具执行失败",
                error_code="internal_error",
            )

        envelope = build_tool_message_content(
            tool_call, result, settings.tool_result_max_chars
        )
        await asyncio.to_thread(persist_tool, envelope, tool_call_id)
        messages.append(
            ToolMessage(content=result.content, tool_call_id=tool_call_id)
        )

        async for chunk in model.astream(messages):
            text = extract_text(chunk)
            if text:
                pending_text += text
                yield format_sse({"delta": text})

        # 工具路径始终落一条最终 assistant 行（空文本时 content 为 None），
        # 保证成功的工具轮次固定是 user/assistant/tool/assistant 四行。
        await asyncio.to_thread(persist_assistant, pending_text)
        pending_text = ""

        yield format_sse(
            {
                "done": True,
                "conversation_id": conversation_id,
                "trimmed": trimmed,
                "tool_calls": [tool_call.get("name")],
            }
        )
    except Exception:
        logger.exception("chat stream failed for conversation %s", conversation_id)
        if pending_text:
            try:
                await asyncio.to_thread(persist_assistant, pending_text)
            except Exception:
                logger.exception(
                    "failed to persist partial assistant text for conversation %s",
                    conversation_id,
                )
        yield format_sse(
            {
                "code": "upstream_error",
                "message": "模型服务暂时不可用，请稍后重试",
            },
            event="error",
        )

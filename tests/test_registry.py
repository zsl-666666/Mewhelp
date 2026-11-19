import asyncio
import json
from typing import Annotated, Literal

import pytest
from langchain_core.tools import InjectedToolArg, tool

from app.tools.registry import ToolRegistry


@tool
def echo(text: str) -> str:
    """回显文本。"""
    return json.dumps({"ok": True, "data": {"text": text}}, ensure_ascii=False)


@tool
async def slow(text: str) -> str:
    """故意慢。"""
    await asyncio.sleep(0.2)
    return json.dumps({"ok": True, "data": {"text": text}}, ensure_ascii=False)


@tool
def boom(text: str) -> str:
    """故意失败。"""
    raise RuntimeError("boom")


@tool
def needs_int(value: int) -> str:
    """需要整数。"""
    return json.dumps({"ok": True, "data": {"value": value}}, ensure_ascii=False)


@tool
def with_conversation(
    text: str,
    conversation_id: Annotated[int, InjectedToolArg],
) -> str:
    """需要注入会话 id。"""
    return json.dumps({"ok": True, "data": {"conversation_id": conversation_id}}, ensure_ascii=False)


def make_flaky(failures: int):
    calls = {"count": 0}

    @tool
    def flaky(text: str) -> str:
        """前几次失败，之后成功。"""
        calls["count"] += 1
        if calls["count"] <= failures:
            raise RuntimeError("flaky")
        return json.dumps({"ok": True, "data": {"text": text}}, ensure_ascii=False)

    return flaky


def build_registry(tool_obj, *, retryable=True):
    registry = ToolRegistry()
    registry.register(tool_obj, label="测试工具", retryable=retryable)
    return registry


async def test_execute_stream_success():
    registry = build_registry(echo)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "echo", "args": {"text": "hi"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "completed"]
    assert events[-1].result.ok is True
    assert json.loads(events[-1].result.content)["data"]["text"] == "hi"


async def test_execute_stream_validation_error_does_not_retry():
    registry = build_registry(needs_int)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "needs_int", "args": {"value": "bad"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "validation_error"
    assert events[-1].result.attempts == 0
    assert json.loads(events[-1].result.content) == {
        "ok": False,
        "error": "参数校验失败",
    }


async def test_execute_stream_timeout_retries_then_timeout():
    registry = build_registry(slow)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "slow", "args": {"text": "hi"}},
            timeout_seconds=0.01,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == [
        "started", "retrying", "retrying", "timeout"
    ]
    assert events[-1].result.error_code == "timeout"
    assert events[-1].result.attempts == 3
    assert json.loads(events[-1].result.content) == {
        "ok": False,
        "error": "工具执行超时",
    }


async def test_execute_stream_non_retryable_fails_once():
    registry = build_registry(boom, retryable=False)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "boom", "args": {"text": "hi"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "execution_error"
    assert events[-1].result.attempts == 1
    assert json.loads(events[-1].result.content) == {
        "ok": False,
        "error": "工具执行失败",
    }


async def test_unknown_tool_fails_without_retry():
    registry = build_registry(echo)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "missing", "args": {}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "unknown_tool"
    assert events[-1].result.attempts == 0
    assert json.loads(events[-1].result.content) == {
        "ok": False,
        "error": "未知工具: missing",
    }


async def test_injected_arg_hidden_from_model_schema():
    registry = build_registry(with_conversation)
    schema = registry.bindable_tools()[0].tool_call_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]

    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "with_conversation", "args": {"text": "hi"}},
            injected_args={"conversation_id": 7},
            timeout_seconds=1,
            max_retries=0,
            retry_delay_seconds=0,
        )
    ]
    assert events[-1].result.ok is True
    assert events[-1].arguments == {"text": "hi"}


async def test_execute_stream_retries_execution_error_then_succeeds():
    registry = build_registry(make_flaky(failures=1))
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "flaky", "args": {"text": "hi"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "retrying", "completed"]
    assert events[1].attempt == 2
    assert events[-1].result.ok is True
    assert events[-1].result.attempts == 2


async def test_execute_returns_final_result():
    registry = build_registry(echo)
    result = await registry.execute(
        {"id": "call_1", "name": "echo", "args": {"text": "hi"}},
        timeout_seconds=1,
        max_retries=0,
        retry_delay_seconds=0,
    )
    assert result.ok is True
    assert result.attempts == 1
    assert result.duration_ms >= 0
    assert json.loads(result.content)["data"]["text"] == "hi"

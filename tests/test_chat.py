import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from sqlalchemy import select

from app.models import Message
from app.repositories.conversations import append_message, create_conversation
from app.services.chat import build_tool_message_content, stream_chat
from app.tools.ecommerce import query_logistics
from app.tools.registry import ToolExecutionResult, ToolRegistry
from tests.fakes import FakeStreamModel, FakeToolCallModel, make_settings


@tool
def broken_lookup(order_id: str) -> str:
    """总是失败的工具，用于验证失败收敛。"""
    raise RuntimeError("boom")


def test_build_tool_message_content_truncates_at_limit():
    tool_call = {
        "id": "call_1",
        "name": "query_logistics",
        "args": {"order_id": "1001"},
    }
    result = ToolExecutionResult(
        ok=True,
        content="x" * 5000,
        error=None,
        error_code=None,
        retryable=False,
        attempts=1,
        duration_ms=12,
    )
    payload = json.loads(build_tool_message_content(tool_call, result, max_chars=4000))
    assert payload["truncated"] is True
    assert payload["original_chars"] == 5000
    assert len(payload["result"]) == 4000
    assert payload["tool"] == "query_logistics"
    assert payload["arguments"] == {"order_id": "1001"}


def test_build_tool_message_content_failure_envelope():
    tool_call = {
        "id": "call_1",
        "name": "query_logistics",
        "args": {"order_id": "1001"},
    }
    result = ToolExecutionResult(
        ok=False,
        content='{"ok": false, "error": "工具执行超时"}',
        error="工具执行超时",
        error_code="timeout",
        retryable=True,
        attempts=3,
        duration_ms=5003,
    )
    payload = json.loads(build_tool_message_content(tool_call, result, max_chars=4000))
    assert payload["ok"] is False
    assert payload["error_code"] == "timeout"
    assert payload["retryable"] is True
    assert payload["attempts"] == 3
    assert payload["duration_ms"] == 5003


async def test_stream_chat_no_tool_path_persists_two_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="你好")

    model = FakeToolCallModel(final_chunks=["你", "好"])
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=ToolRegistry(),
            model=model,
            settings=make_settings(),
        )
    ]

    assert frames[0] == 'event: meta\ndata: {"conversation_id": 1, "user_id": "u1"}\n\n'
    assert any('"delta": "你"' in frame for frame in frames)
    assert any('"done": true' in frame and '"tool_calls": []' in frame for frame in frames)
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[-1].content == "你好"


async def test_stream_chat_tool_path_persists_four_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")

    registry = ToolRegistry()
    registry.register(query_logistics, label="查物流")
    model = FakeToolCallModel(
        tool_calls=[{
            "name": "query_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["预计明天送达。"],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert any('"status": "running"' in frame for frame in frames)
    assert any('"status": "completed"' in frame for frame in frames)
    assert any('"tool_calls": ["query_logistics"]' in frame for frame in frames)
    tool_messages = [m for m in model.final_messages if isinstance(m, ToolMessage)]
    assert tool_messages and json.loads(tool_messages[0].content)["ok"] is True
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["id"] == "call_1"
    assert rows[2].tool_call_id == "call_1"
    envelope = json.loads(rows[2].content)
    assert envelope["tool"] == "query_logistics"
    assert envelope["arguments"] == {"order_id": "1001"}
    assert envelope["ok"] is True


async def test_stream_chat_only_first_tool_call(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="查订单和物流")

    registry = ToolRegistry()
    registry.register(query_logistics, label="查物流")
    model = FakeToolCallModel(
        tool_calls=[
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "query_order",
                "args": {"order_id": "1001"},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
        final_chunks=["已为您查询。"],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert any('"tool_calls": ["query_logistics"]' in frame for frame in frames)
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert len(rows[1].tool_calls) == 1
    assert rows[1].tool_calls[0]["id"] == "call_1"


async def test_stream_chat_model_error_emits_error_event(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="你好")

    model = FakeStreamModel(["你"], error_after=1)
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=ToolRegistry(),
            model=model,
            settings=make_settings(),
        )
    ]

    assert any("event: error" in frame for frame in frames)
    assert any('"delta": "你"' in frame for frame in frames)


async def test_stream_chat_tool_path_persists_final_row_when_text_empty(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")

    registry = ToolRegistry()
    registry.register(query_logistics, label="查物流")
    model = FakeToolCallModel(
        tool_calls=[{
            "name": "query_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=[],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert any('"done": true' in frame for frame in frames)
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[-1].content is None


async def test_stream_chat_tool_failure_converges_without_error_event(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")

    registry = ToolRegistry()
    registry.register(broken_lookup, label="坏工具", retryable=False)
    model = FakeToolCallModel(
        tool_calls=[{
            "name": "broken_lookup",
            "args": {"order_id": "1001"},
            "id": "call_fail",
            "type": "tool_call",
        }],
        final_chunks=["抱歉，", "工具暂时不可用。"],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert not any("event: error" in frame for frame in frames)
    assert any('"status": "failed"' in frame for frame in frames)
    assert any('"delta": "抱歉，"' in frame for frame in frames)
    assert any(
        '"done": true' in frame and '"tool_calls": ["broken_lookup"]' in frame
        for frame in frames
    )

    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    envelope = json.loads(rows[2].content)
    assert envelope["ok"] is False
    assert envelope["error_code"] == "execution_error"
    envelope_result = json.loads(envelope["result"])
    assert envelope_result["ok"] is False
    assert envelope_result["error"]

    tool_messages = [m for m in model.final_messages if isinstance(m, ToolMessage)]
    assert tool_messages
    assert tool_messages[0].content
    tool_payload = json.loads(tool_messages[0].content)
    assert tool_payload["ok"] is False
    assert tool_payload["error"]

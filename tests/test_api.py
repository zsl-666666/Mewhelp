import asyncio
import json

from langchain_core.tools import tool
from sqlalchemy import select

from app.config import get_settings
from app.db import get_db_session_factory
from app.llm import get_chat_model
from app.main import app, get_tool_registry
from app.models import Conversation, Message
from app.repositories.conversations import append_message, create_conversation
from app.tools.registry import ToolRegistry
from tests.fakes import (
    FakeStreamModel,
    FakeStructuredModel,
    FakeToolCallModel,
    make_settings,
)


def override_model(model):
    app.dependency_overrides[get_chat_model] = lambda: model


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "test-model"}


def test_chat_creates_conversation_and_returns_meta(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(final_chunks=["你好"]))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "你好"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["x-accel-buffering"] == "no"
        body = "".join(response.iter_text())

    assert "event: meta" in body
    assert '"conversation_id": 1' in body
    assert '"trimmed": false' in body
    assert '"done": true' in body
    with session_factory() as session:
        convo = session.scalar(select(Conversation))
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert convo.status == "进行中"
    assert [row.role for row in rows] == ["user", "assistant"]


def test_chat_continues_existing_conversation(client, session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="第一轮")
        append_message(session, convo.id, role="assistant", content="第一轮回复")

    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    model = FakeToolCallModel(final_chunks=["第二轮回复"])
    override_model(model)
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "conversation_id": convo.id, "message": "第二轮"},
    ) as response:
        body = "".join(response.iter_text())

    assert f'"conversation_id": {convo.id}' in body
    assert any("第一轮" in str(message.content) for message in model.final_messages)
    assert any("第二轮" in str(message.content) for message in model.final_messages)


def test_chat_unknown_conversation_returns_404(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(final_chunks=["不应调用"]))
    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "conversation_id": 99999, "message": "你好"},
    )
    assert response.status_code == 404


def test_chat_rejects_old_messages_payload_and_blank_message(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    old = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    blank = client.post("/api/chat", json={"user_id": "u1", "message": "   "})
    assert old.status_code == 422
    assert blank.status_code == 422


def test_chat_upstream_failure_emits_error_frame_and_keeps_user_message(
    client, session_factory
):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    # 空注册表让模型无需 bind_tools，直接覆盖上游模型抛错路径。
    app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()
    override_model(FakeStreamModel([], error_after=0))
    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "message": "你好"},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: error" in body
    assert "upstream_error" in body
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user"]
    assert rows[0].content == "你好"


def test_chat_tool_completed_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "query_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["预计明天送达。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: tool" in body
    assert '"status": "running"' in body
    assert '"status": "completed"' in body
    assert '"tool_calls": ["query_logistics"]' in body
    assert "event: error" not in body


@tool
async def slow_logistics(order_id: str) -> str:
    """慢查询，用于超时测试。"""
    await asyncio.sleep(0.2)
    return json.dumps({"ok": True, "data": {"order_id": order_id}})


@tool
def broken_logistics(order_id: str) -> str:
    """总是失败，用于错误帧测试。"""
    raise RuntimeError("boom")


def override_registry(tool_obj):
    registry = ToolRegistry()
    registry.register(tool_obj, label="查物流")
    app.dependency_overrides[get_tool_registry] = lambda: registry


def test_chat_tool_timeout_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    app.dependency_overrides[get_settings] = lambda: make_settings(
        tool_timeout_seconds=0.01,
        tool_retry_delay_seconds=0,
    )
    override_registry(slow_logistics)
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "slow_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["物流暂时查不到，请稍后再试。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert '"status": "retrying"' in body
    assert '"status": "timeout"' in body
    assert '"error_code": "timeout"' in body
    assert "event: error" not in body


def test_chat_tool_failure_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_registry(broken_logistics)
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "broken_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["查询失败，已为您转接人工。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert '"status": "failed"' in body
    assert '"error_code": "execution_error"' in body
    assert '"tool_calls": ["broken_logistics"]' in body
    assert "event: error" not in body


def test_extract_returns_structured_json(client):
    fake = FakeStructuredModel(
        result={"order_id": "A1", "request_type": "退货", "expected_solution": "退货退款"}
    )

    def override_model():
        return fake

    app.dependency_overrides[get_chat_model] = override_model
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 收到破损，我要退货退款"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A1",
        "request_type": "退货",
        "expected_solution": "退货退款",
    }


def test_extract_returns_502_on_failure(client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(
        error=RuntimeError("boom")
    )
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 要退货"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "结构化提取失败，请稍后重试"


def test_extract_returns_502_on_validation_failure(client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(
        result={"request_type": "not-a-type"}
    )
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 要退货"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "结构化提取失败，请稍后重试"


def test_extract_rejects_blank_text(client):
    fake = FakeStructuredModel(
        result={"order_id": None, "request_type": "咨询", "expected_solution": None}
    )
    app.dependency_overrides[get_chat_model] = lambda: fake
    response = client.post("/api/after-sales/extract", json={"text": "   "})
    assert response.status_code == 422
    assert fake.calls == []
    assert fake.runnable is None

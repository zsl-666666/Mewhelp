import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.repositories.conversations import (
    append_message,
    create_conversation,
    load_history,
)
from app.repositories.faq import search_faq
from app.repositories.tickets import create_ticket_record
from app.models import Faq


def test_load_history_rebuilds_tool_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")
        append_message(
            session,
            convo.id,
            role="assistant",
            content="",
            tool_calls=[{
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        envelope = json.dumps({
            "tool": "query_logistics",
            "arguments": {"order_id": "1001"},
            "ok": True,
            "result": '{"ok": true, "data": {"status": "运输中"}}',
            "truncated": False,
            "original_chars": 40,
            "attempts": 1,
            "duration_ms": 12,
        }, ensure_ascii=False)
        append_message(
            session,
            convo.id,
            role="tool",
            content=envelope,
            tool_call_id="call_1",
        )
        append_message(session, convo.id, role="assistant", content="预计明天送达。")

        history = load_history(session, convo.id)

    assert [type(message).__name__ for message in history] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[1].tool_calls[0]["id"] == "call_1"
    assert isinstance(history[2], ToolMessage)
    assert history[2].tool_call_id == "call_1"
    assert history[2].content == '{"ok": true, "data": {"status": "运输中"}}'


def test_search_faq_escapes_like_wildcards(session_factory):
    with session_factory() as session:
        session.add_all([
            Faq(question="含 100% 纯棉", answer="是", category="商品"),
            Faq(question="100 元优惠券", answer="是", category="活动"),
        ])
        session.commit()
        rows = search_faq(session, "100%")

    assert [row.question for row in rows] == ["含 100% 纯棉"]


def test_create_ticket_record_updates_conversation_status(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        ticket = create_ticket_record(session, convo.id, "耳机坏了", "售后")
        session.refresh(convo)

    assert ticket.status == "待处理"
    assert ticket.ticket_type == "售后"
    assert ticket.ticket_no.startswith("T")
    assert len(ticket.ticket_no) == 19
    assert convo.status == "已转人工"


def test_create_ticket_record_rejects_invalid_type_and_missing_conversation(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        with pytest.raises(ValueError):
            create_ticket_record(session, convo.id, "x", "退款")
        with pytest.raises(ValueError):
            create_ticket_record(session, 99999, "x", "售后")

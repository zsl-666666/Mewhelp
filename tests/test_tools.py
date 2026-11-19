import json

from app.repositories.faq import seed_faq
from app.tools import build_tool_registry
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.knowledge import build_query_faq_tool
from app.tools.tickets import build_create_ticket_tool


def test_ecommerce_tools_are_deterministic():
    first = query_order.invoke({"order_id": "1001"})
    second = query_order.invoke({"order_id": "1001"})
    assert first == second
    assert json.loads(first)["ok"] is True

    logistics = json.loads(query_logistics.invoke({"order_id": "1001"}))
    product = json.loads(query_product.invoke({"product_id": "P100"}))
    assert logistics["data"]["order_id"] == "1001"
    assert product["data"]["product_id"] == "P100"


def test_query_faq_tool_hit_and_miss(session_factory):
    with session_factory() as session:
        seed_faq(session)

    query_faq = build_query_faq_tool(session_factory)
    hit = json.loads(query_faq.invoke({"keyword": "退货政策"}))
    miss = json.loads(query_faq.invoke({"keyword": "邮费"}))

    assert hit["ok"] is True
    assert any("退货政策" in item["question"] for item in hit["data"])
    assert miss["ok"] is True
    assert miss["data"] == []
    assert miss["message"] == "未找到相关 FAQ"


def test_create_ticket_tool_writes_and_updates_status(session_factory):
    from app.repositories.conversations import create_conversation

    with session_factory() as session:
        convo = create_conversation(session, "u1")

    create_ticket = build_create_ticket_tool(session_factory)
    result = json.loads(create_ticket.invoke({
        "description": "耳机坏了，需要人工处理",
        "ticket_type": "售后",
        "conversation_id": convo.id,
    }))

    assert result["ok"] is True
    assert result["data"]["ticket_type"] == "售后"
    with session_factory() as session:
        from app.models import Conversation, Ticket
        assert session.get(Conversation, convo.id).status == "已转人工"
        assert session.get(Ticket, result["data"]["ticket_no"]).status == "待处理"


def test_build_tool_registry_has_five_tools_and_labels(session_factory):
    registry = build_tool_registry(session_factory)
    assert [tool.name for tool in registry.bindable_tools()] == [
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket"
    ]
    assert registry.get("query_logistics").label == "查物流"
    assert registry.get("create_ticket").retryable is False


def test_create_ticket_schema_hides_injected_conversation_id(session_factory):
    registry = build_tool_registry(session_factory)
    schema = registry.get("create_ticket").tool.tool_call_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]
    assert "ticket_type" in schema["properties"]

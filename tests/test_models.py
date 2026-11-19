import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Conversation, Faq, Message, Ticket


def test_models_roundtrip_and_nullable_tool_fields(session_factory):
    with session_factory() as session:
        convo = Conversation(user_id="u1", status="进行中")
        session.add(convo)
        session.commit()
        session.refresh(convo)

        message = Message(
            conversation_id=convo.id,
            role="assistant",
            content=None,
            tool_calls=[{
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        faq = Faq(
            question="退货政策是什么？",
            answer="签收后 7 天内支持无理由退货。",
            category="售后",
        )
        ticket = Ticket(
            ticket_no="T20260701008",
            conversation_id=convo.id,
            description="耳机坏了",
            ticket_type="售后",
        )
        session.add_all([message, faq, ticket])
        session.commit()
        session.refresh(faq)

        assert convo.status == "进行中"
        assert message.content is None
        assert message.tool_calls[0]["id"] == "call_1"
        assert faq.updated_at is not None
        assert ticket.status == "待处理"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Conversation(user_id="u1", status="active"),
        lambda: Message(conversation_id=1, role="system", content="x"),
        lambda: Ticket(
            ticket_no="T1",
            conversation_id=1,
            description="x",
            ticket_type="退款",
        ),
    ],
)
def test_models_reject_invalid_enum_values(session_factory, factory):
    with session_factory() as session:
        session.add(factory())
        with pytest.raises(IntegrityError):
            session.commit()

import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesExtraction, ChatRequest


def test_chat_request_defaults():
    request = ChatRequest(message="你好")
    assert request.user_id == "demo-user"
    assert request.conversation_id is None


def test_chat_request_accepts_conversation_id():
    request = ChatRequest(user_id="u1", conversation_id=12, message="继续")
    assert request.conversation_id == 12


@pytest.mark.parametrize("payload", [
    {"message": "   "},
    {"user_id": "   ", "message": "你好"},
    {"user_id": "u1", "conversation_id": 0, "message": "你好"},
])
def test_chat_request_rejects_invalid(payload):
    with pytest.raises(ValidationError):
        ChatRequest(**payload)


def test_after_sales_extraction_accepts_nulls():
    result = AfterSalesExtraction(
        order_id=None,
        request_type="咨询",
        expected_solution=None,
    )
    assert result.order_id is None


def test_after_sales_extraction_rejects_unknown_type():
    with pytest.raises(ValidationError):
        AfterSalesExtraction(
            order_id="A1",
            request_type="投诉",
            expected_solution="道歉",
        )


def test_after_sales_extraction_normalizes_blank_strings():
    result = AfterSalesExtraction(
        order_id="  ",
        request_type="咨询",
        expected_solution="",
    )
    assert result.order_id is None
    assert result.expected_solution is None

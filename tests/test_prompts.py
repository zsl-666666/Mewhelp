from langchain_core.messages import SystemMessage

from app.prompts.after_sales import AFTER_SALES_SYSTEM_PROMPT
from app.prompts.customer_service import build_customer_service_prompt


def test_customer_service_prompt_substitutes_variables():
    prompt = build_customer_service_prompt()
    messages = prompt.format_messages(
        company_name="测试商城",
        service_hours="10:00-18:00",
        return_window_days=15,
    )
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "测试商城" in messages[0].content
    assert "10:00-18:00" in messages[0].content
    assert "15" in messages[0].content


def test_after_sales_prompt_lists_all_request_types():
    for value in ("退款", "退货", "换货", "维修", "物流", "咨询", "其他"):
        assert value in AFTER_SALES_SYSTEM_PROMPT
    assert "禁止猜测" in AFTER_SALES_SYSTEM_PROMPT

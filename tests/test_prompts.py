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


from app.prompts.customer_service import TOOL_USAGE_RULES


def test_tool_usage_rules_pin_single_call_and_faq_keyword():
    assert "只调用一个工具" in TOOL_USAGE_RULES
    assert "邮费" in TOOL_USAGE_RULES
    assert "运费" in TOOL_USAGE_RULES
    assert "不要改写" in TOOL_USAGE_RULES
    assert "create_ticket" in TOOL_USAGE_RULES
    for value in ("售后", "投诉", "咨询"):
        assert value in TOOL_USAGE_RULES

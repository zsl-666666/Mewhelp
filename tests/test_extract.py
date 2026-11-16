from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.after_sales import AFTER_SALES_SYSTEM_PROMPT
from app.schemas import AfterSalesExtraction
from app.services.extract import extract_after_sales
from tests.fakes import FakeStructuredModel, make_settings


async def test_extract_uses_schema_and_configured_method():
    expected = AfterSalesExtraction(
        order_id="A1", request_type="退货", expected_solution="退货退款"
    )
    model = FakeStructuredModel(result=expected)
    result = await extract_after_sales("订单 A1 要退货", model, make_settings())
    assert result == expected
    assert model.calls == [(AfterSalesExtraction, "function_calling")]


async def test_extract_passes_configured_method():
    model = FakeStructuredModel(
        result={"order_id": None, "request_type": "咨询", "expected_solution": None}
    )
    await extract_after_sales(
        "请问支持退货吗",
        model,
        make_settings(structured_output_method="json_schema"),
    )
    assert model.calls == [(AfterSalesExtraction, "json_schema")]


async def test_extract_accepts_dict_result():
    model = FakeStructuredModel(
        result={"order_id": None, "request_type": "咨询", "expected_solution": None}
    )
    result = await extract_after_sales("请问支持退货吗", model, make_settings())
    assert result == AfterSalesExtraction(
        order_id=None, request_type="咨询", expected_solution=None
    )


async def test_extract_builds_system_and_user_messages():
    model = FakeStructuredModel(
        result={"order_id": None, "request_type": "咨询", "expected_solution": None}
    )
    await extract_after_sales("请问支持退货吗", model, make_settings())
    assert isinstance(model.runnable.messages[0], SystemMessage)
    assert model.runnable.messages[0].content == AFTER_SALES_SYSTEM_PROMPT
    assert isinstance(model.runnable.messages[1], HumanMessage)
    assert model.runnable.messages[1].content == "请问支持退货吗"

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.prompts.after_sales import AFTER_SALES_SYSTEM_PROMPT
from app.schemas import AfterSalesExtraction


async def extract_after_sales(
    text: str,
    model: BaseChatModel,
    settings: Settings,
) -> AfterSalesExtraction:
    structured = model.with_structured_output(
        AfterSalesExtraction,
        method=settings.structured_output_method,
    )
    result = await structured.ainvoke(
        [
            SystemMessage(content=AFTER_SALES_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
    )
    if isinstance(result, AfterSalesExtraction):
        return result
    return AfterSalesExtraction.model_validate(result)

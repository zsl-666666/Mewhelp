import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.schemas import AfterSalesExtraction, AfterSalesRequest, ChatRequest
from app.services.chat import stream_chat
from app.services.extract import extract_after_sales

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 缺少必填配置时在启动阶段直接失败，而不是等到首个请求才报 500。
    get_settings()
    yield


app = FastAPI(title="电商智能客服", lifespan=lifespan)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    model: BaseChatModel = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
):
    history = [HumanMessage(content=request.message)]
    return StreamingResponse(
        stream_chat(history, model, settings),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/after-sales/extract", response_model=AfterSalesExtraction)
async def after_sales_extract(
    request: AfterSalesRequest,
    model: BaseChatModel = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
) -> AfterSalesExtraction:
    try:
        return await extract_after_sales(request.text, model, settings)
    except Exception:
        logger.exception("structured extraction failed")
        raise HTTPException(
            status_code=502,
            detail="结构化提取失败，请稍后重试",
        )

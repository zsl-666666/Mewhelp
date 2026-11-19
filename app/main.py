import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import get_db_session_factory
from app.llm import get_chat_model
from app.repositories.conversations import (
    append_message,
    create_conversation,
    get_conversation,
)
from app.schemas import AfterSalesExtraction, AfterSalesRequest, ChatRequest
from app.services.chat import stream_chat
from app.services.extract import extract_after_sales
from app.tools import build_tool_registry
from app.tools.registry import ToolRegistry

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


def get_tool_registry(
    session_factory: sessionmaker[Session] = Depends(get_db_session_factory),
) -> ToolRegistry:
    return build_tool_registry(session_factory)


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    model: BaseChatModel = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
    session_factory: sessionmaker[Session] = Depends(get_db_session_factory),
    registry: ToolRegistry = Depends(get_tool_registry),
):
    with session_factory() as session:
        if request.conversation_id is not None:
            conversation = get_conversation(session, request.conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            conversation = create_conversation(session, request.user_id)
        append_message(
            session,
            conversation.id,
            role="user",
            content=request.message,
        )
        conversation_id = conversation.id

    return StreamingResponse(
        stream_chat(
            conversation_id=conversation_id,
            user_id=request.user_id,
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=settings,
        ),
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

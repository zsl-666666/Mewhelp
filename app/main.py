from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings
from app.llm import get_chat_model
from app.schemas import ChatRequest
from app.services.chat import stream_chat, to_lc_message

app = FastAPI(title="电商智能客服")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    model: BaseChatModel = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
):
    history = [to_lc_message(message) for message in request.messages]
    return StreamingResponse(
        stream_chat(history, model, settings),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

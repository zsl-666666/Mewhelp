from fastapi import Depends, FastAPI

from app.config import Settings, get_settings

app = FastAPI(title="电商智能客服")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}

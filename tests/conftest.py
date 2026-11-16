import os

import pytest
from fastapi.testclient import TestClient

# app.main 的 lifespan 会在启动时解析配置；离线测试需要先提供必填环境变量。
os.environ.setdefault("LLM_BASE_URL", "https://example.com/v1")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "test-model")

from app.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_base_url="https://example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        _env_file=None,
    )


@pytest.fixture
def client(test_settings):
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

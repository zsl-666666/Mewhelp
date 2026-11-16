import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


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

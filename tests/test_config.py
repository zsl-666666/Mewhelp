import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    settings = Settings(_env_file=None)
    assert settings.context_token_budget == 3000
    assert settings.structured_output_method == "function_calling"


def test_missing_required_settings_raises(monkeypatch):
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

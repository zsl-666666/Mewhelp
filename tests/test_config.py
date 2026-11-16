import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_json_mode_is_not_a_supported_structured_output_method(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, structured_output_method="json_mode")


def test_app_startup_fails_without_required_settings(tmp_path):
    env = os.environ.copy()
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO_ROOT)

    script = (
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "TestClient(app).__enter__()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "llm_base_url" in (result.stdout + result.stderr).lower()

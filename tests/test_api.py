from app.llm import get_chat_model
from app.main import app
from tests.fakes import FakeStreamModel


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "test-model"}


def test_chat_returns_sse_stream(client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStreamModel(["你", "好"])
    with client.stream(
        "POST",
        "/api/chat",
        json={"messages": [{"role": "user", "content": "你好"}]},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["x-accel-buffering"] == "no"
        body = "".join(response.iter_text())
    assert 'data: {"delta": "你"}' in body
    assert 'data: {"done": true, "trimmed": false}' in body


def test_chat_rejects_trailing_assistant(client):
    response = client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "您好"},
            ]
        },
    )
    assert response.status_code == 422

from app.llm import get_chat_model
from app.main import app
from tests.fakes import FakeStreamModel, FakeStructuredModel


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


def test_extract_returns_structured_json(client):
    fake = FakeStructuredModel(
        result={"order_id": "A1", "request_type": "退货", "expected_solution": "退货退款"}
    )

    def override_model():
        return fake

    app.dependency_overrides[get_chat_model] = override_model
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 收到破损，我要退货退款"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A1",
        "request_type": "退货",
        "expected_solution": "退货退款",
    }


def test_extract_returns_502_on_failure(client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(
        error=RuntimeError("boom")
    )
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 要退货"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "结构化提取失败，请稍后重试"


def test_extract_returns_502_on_validation_failure(client):
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredModel(
        result={"request_type": "not-a-type"}
    )
    response = client.post(
        "/api/after-sales/extract",
        json={"text": "订单 A1 要退货"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "结构化提取失败，请稍后重试"


def test_extract_rejects_blank_text(client):
    fake = FakeStructuredModel(
        result={"order_id": None, "request_type": "咨询", "expected_solution": None}
    )
    app.dependency_overrides[get_chat_model] = lambda: fake
    response = client.post("/api/after-sales/extract", json={"text": "   "})
    assert response.status_code == 422
    assert fake.calls == []
    assert fake.runnable is None

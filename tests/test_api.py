"""Tests for the SparkLLM FastAPI server."""
import pytest
import torch
from httpx import AsyncClient, ASGITransport
from inference.server import app, _model, _tokenizer, _device
import inference.server as server_module
from inference.model import load_model, load_tokenizer


@pytest.fixture(scope="module", autouse=True)
def load_model_for_tests():
    """Load the model once for all API tests (lifespan doesn't run in test transport)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_module._device = device
    server_module._model = load_model(device)
    server_module._tokenizer = load_tokenizer()
    yield
    server_module._model = None
    server_module._tokenizer = None
    server_module._device = None


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Health endpoint ---

@pytest.mark.anyio
async def test_health(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# --- Generate endpoint ---

@pytest.mark.anyio
async def test_generate_basic(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "The knight",
        "max_tokens": 10,
        "temperature": 0.8,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "generated_text" in data
    assert len(data["generated_text"]) > len("The knight")
    assert data["prompt"] == "The knight"


@pytest.mark.anyio
async def test_generate_with_params(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "Once upon a time",
        "max_tokens": 20,
        "temperature": 0.5,
        "top_k": 20,
        "top_p": 0.85,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["generated_text"], str)


@pytest.mark.anyio
async def test_generate_empty_prompt_rejected(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "",
        "max_tokens": 10,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_generate_invalid_temperature(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_generate_max_tokens_limit(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 5000,
    })
    assert resp.status_code == 422


# --- CORS ---

@pytest.mark.anyio
async def test_cors_headers(async_client):
    resp = await async_client.options(
        "/generate",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


# --- Response schema ---

@pytest.mark.anyio
async def test_response_has_device_field(async_client):
    resp = await async_client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "device" in data
    assert data["device"] in ("cuda", "cpu")


# --- Streaming endpoint ---

@pytest.mark.anyio
async def test_generate_stream(async_client):
    resp = await async_client.post("/generate/stream", json={
        "prompt": "The knight",
        "max_tokens": 10,
        "temperature": 0.8,
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    import json
    lines = resp.text.strip().split("\n")
    events = [json.loads(l.removeprefix("data: ")) for l in lines if l.startswith("data: ")]
    assert len(events) >= 2  # at least prompt + done
    assert "token" in events[0]  # first event is the prompt
    assert events[-1].get("done") is True  # last event signals completion

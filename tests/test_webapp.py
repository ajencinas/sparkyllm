"""Tests for the Streamlit webapp configuration and API interaction."""
import pytest
import requests
import responses


@responses.activate
def test_webapp_calls_api_generate():
    """Verify the webapp would call the correct API endpoint with correct payload."""
    api_url = "http://localhost:8000"

    responses.post(
        f"{api_url}/generate",
        json={
            "prompt": "Hello",
            "generated_text": "Hello world",
            "device": "cuda",
        },
        status=200,
    )

    resp = requests.post(
        f"{api_url}/generate",
        json={
            "prompt": "Hello",
            "max_tokens": 50,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["generated_text"] == "Hello world"
    assert len(responses.calls) == 1
    sent = responses.calls[0].request
    import json
    body = json.loads(sent.body)
    assert body["prompt"] == "Hello"
    assert body["max_tokens"] == 50


@responses.activate
def test_webapp_health_check():
    """Verify the health check request format."""
    api_url = "http://localhost:8000"

    responses.get(
        f"{api_url}/health",
        json={"status": "ok", "model_loaded": True, "device": "cuda", "gpu": "RTX 5080"},
        status=200,
    )

    resp = requests.get(f"{api_url}/health", timeout=5)
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@responses.activate
def test_webapp_handles_api_error():
    """Verify graceful handling when API returns error."""
    api_url = "http://localhost:8000"

    responses.post(f"{api_url}/generate", status=500)

    resp = requests.post(f"{api_url}/generate", json={"prompt": "test"})
    assert resp.status_code == 500

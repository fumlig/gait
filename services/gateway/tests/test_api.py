"""Tests for the API gateway.

These tests mock the httpx client to avoid needing real backend services.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHATTERBOX_MODELS = {
    "object": "list",
    "data": [
        {"id": "chatterbox-turbo", "object": "model", "created": 0, "owned_by": "resemble-ai"}
    ],
}

WHISPERX_MODELS = {
    "object": "list",
    "data": [
        {"id": "whisper-1", "object": "model", "created": 0, "owned_by": "whisperx"},
        {"id": "large-v3", "object": "model", "created": 0, "owned_by": "whisperx"},
    ],
}


def _mock_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b"",
    content_type: str = "application/json",
) -> httpx.Response:
    """Create a mock httpx Response."""
    if json_data is not None:
        import json

        content = json.dumps(json_data).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": content_type},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client():
    """Patch the httpx client used by the proxy module."""
    client = AsyncMock(spec=httpx.AsyncClient)
    with patch("gateway_service.proxy._client", client):
        yield client


@pytest.fixture()
def app(mock_client):
    """Create a test app that skips the real lifespan."""
    from fastapi import FastAPI

    from gateway_service.routes import audio, health, models

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(audio.router)
    test_app.include_router(models.router)
    test_app.include_router(health.router)
    return test_app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_all_healthy(client: AsyncClient, mock_client: AsyncMock):
    """Gateway reports 'ok' when all backends are healthy."""
    mock_client.get.return_value = _mock_httpx_response(
        json_data={"status": "ok", "model_loaded": True}
    )
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "chatterbox" in data["backends"]
    assert "whisperx" in data["backends"]


async def test_health_backend_down(client: AsyncClient, mock_client: AsyncMock):
    """Gateway reports 'degraded' when a backend is unreachable."""
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# Models (merged)
# ---------------------------------------------------------------------------


async def test_list_models_merged(client: AsyncClient, mock_client: AsyncMock):
    """Gateway merges model lists from all backends."""

    async def mock_get(url: str, **kwargs):
        if "chatterbox" in url:
            return _mock_httpx_response(json_data=CHATTERBOX_MODELS)
        if "whisperx" in url:
            return _mock_httpx_response(json_data=WHISPERX_MODELS)
        return _mock_httpx_response(status_code=404)

    mock_client.get.side_effect = mock_get

    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = {m["id"] for m in data["data"]}
    assert "chatterbox-turbo" in model_ids
    assert "whisper-1" in model_ids
    assert "large-v3" in model_ids


async def test_list_models_backend_failure(client: AsyncClient, mock_client: AsyncMock):
    """Gateway still returns models from healthy backends when one fails."""

    async def mock_get(url: str, **kwargs):
        if "chatterbox" in url:
            return _mock_httpx_response(json_data=CHATTERBOX_MODELS)
        raise httpx.ConnectError("Connection refused")

    mock_client.get.side_effect = mock_get

    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = {m["id"] for m in data["data"]}
    assert "chatterbox-turbo" in model_ids
    # whisperx models should be missing since that backend failed
    assert "whisper-1" not in model_ids


# ---------------------------------------------------------------------------
# Audio proxy
# ---------------------------------------------------------------------------


async def test_proxy_speech(client: AsyncClient, mock_client: AsyncMock):
    """Speech requests are proxied to chatterbox."""
    mock_client.request.return_value = _mock_httpx_response(
        content=b"fake-audio-data",
        content_type="audio/mpeg",
    )
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "chatterbox-turbo", "input": "Hello", "voice": "default"},
    )
    assert resp.status_code == 200
    assert resp.content == b"fake-audio-data"
    # Verify the proxy called the right backend
    mock_client.request.assert_called_once()
    call_kwargs = mock_client.request.call_args
    assert "chatterbox" in call_kwargs.kwargs.get(
        "url", call_kwargs.args[1] if len(call_kwargs.args) > 1 else ""
    )


async def test_proxy_transcriptions(client: AsyncClient, mock_client: AsyncMock):
    """Transcription requests are proxied to whisperx."""
    mock_client.request.return_value = _mock_httpx_response(
        json_data={"text": "Hello world"},
    )
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world"


async def test_proxy_translations(client: AsyncClient, mock_client: AsyncMock):
    """Translation requests are proxied to whisperx."""
    mock_client.request.return_value = _mock_httpx_response(
        json_data={"text": "Translated text"},
    )
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Translated text"


async def test_proxy_backend_unavailable(client: AsyncClient, mock_client: AsyncMock):
    """Gateway returns 502 when a backend is unreachable."""
    mock_client.request.side_effect = httpx.ConnectError("Connection refused")
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "chatterbox-turbo", "input": "Hello", "voice": "default"},
    )
    assert resp.status_code == 502

"""Tests for the Chatterbox TTS API endpoints.

These tests mock the engine so they can run without a GPU or model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
DUMMY_WAV = torch.randn(1, SAMPLE_RATE)  # 1 second of noise


@pytest.fixture()
def mock_engine():
    """Patch the engine singleton so no real model is loaded."""
    with patch("chatterbox_service.engine.engine") as eng:
        eng.is_loaded = True
        eng.sample_rate = SAMPLE_RATE
        eng.list_voices.return_value = ["default"]
        eng.generate.return_value = (DUMMY_WAV, SAMPLE_RATE)
        # Also patch the engine used in routes (same singleton)
        with (
            patch("chatterbox_service.routes.speech.engine", eng),
            patch("chatterbox_service.routes.health.engine", eng),
        ):
            yield eng


@pytest.fixture()
def app(mock_engine):
    """Create a test app that skips the real lifespan (model loading)."""
    from collections.abc import AsyncIterator  # noqa: TC003 — runtime use in return annotation
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from chatterbox_service.routes import health, models, speech

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(speech.router)
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


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


async def test_list_models(client: AsyncClient):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "chatterbox-turbo"


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------


async def test_speech_mp3(client: AsyncClient, mock_engine: MagicMock):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello world",
            "voice": "default",
            "response_format": "mp3",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0
    mock_engine.generate.assert_called_once()


async def test_speech_wav(client: AsyncClient, mock_engine: MagicMock):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello world",
            "voice": "default",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # WAV files start with "RIFF"
    assert resp.content[:4] == b"RIFF"


async def test_speech_unknown_model(client: AsyncClient):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "nonexistent",
            "input": "Hello",
            "voice": "default",
        },
    )
    assert resp.status_code == 400


async def test_speech_unknown_voice(client: AsyncClient):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello",
            "voice": "nonexistent",
        },
    )
    assert resp.status_code == 400


async def test_speech_speed_out_of_range(client: AsyncClient):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello",
            "voice": "default",
            "speed": 5.0,
        },
    )
    assert resp.status_code == 422  # validation error

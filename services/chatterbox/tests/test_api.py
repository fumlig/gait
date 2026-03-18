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
        eng.loaded_models = {
            "chatterbox-turbo": True,
            "chatterbox": False,
            "chatterbox-multilingual": False,
        }
        eng.list_voices.return_value = ["default"]
        eng.generate.return_value = (DUMMY_WAV, SAMPLE_RATE)
        eng.ensure_model.return_value = "chatterbox-turbo"
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


async def test_health_shows_per_model_status(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert data["models"]["chatterbox-turbo"] is True
    assert data["models"]["chatterbox"] is False
    assert data["models"]["chatterbox-multilingual"] is False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


async def test_list_models(client: AsyncClient):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3
    model_ids = {m["id"] for m in data["data"]}
    assert model_ids == {"chatterbox-turbo", "chatterbox", "chatterbox-multilingual"}


async def test_list_models_owned_by(client: AsyncClient):
    resp = await client.get("/v1/models")
    data = resp.json()
    for model in data["data"]:
        assert model["owned_by"] == "resemble-ai"


# ---------------------------------------------------------------------------
# Speech — basic generation
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


# ---------------------------------------------------------------------------
# Speech — model aliases
# ---------------------------------------------------------------------------


async def test_speech_alias_tts1(client: AsyncClient, mock_engine: MagicMock):
    """tts-1 should be accepted and resolved by ensure_model."""
    mock_engine.ensure_model.return_value = "chatterbox-turbo"
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Alias test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    mock_engine.ensure_model.assert_called_with("tts-1")


async def test_speech_alias_tts1hd(client: AsyncClient, mock_engine: MagicMock):
    """tts-1-hd should be accepted and resolved by ensure_model."""
    mock_engine.ensure_model.return_value = "chatterbox"
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1-hd",
            "input": "Alias test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    mock_engine.ensure_model.assert_called_with("tts-1-hd")


# ---------------------------------------------------------------------------
# Speech — chatterbox (original) model
# ---------------------------------------------------------------------------


async def test_speech_original_model(client: AsyncClient, mock_engine: MagicMock):
    """The original chatterbox model should accept exaggeration and cfg_weight."""
    mock_engine.ensure_model.return_value = "chatterbox"
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox",
            "input": "Original model test",
            "voice": "default",
            "exaggeration": 0.8,
            "cfg_weight": 0.7,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["model_name"] == "chatterbox"
    assert call_kwargs.kwargs["exaggeration"] == 0.8
    assert call_kwargs.kwargs["cfg_weight"] == 0.7


# ---------------------------------------------------------------------------
# Speech — multilingual model
# ---------------------------------------------------------------------------


async def test_speech_multilingual(client: AsyncClient, mock_engine: MagicMock):
    """Multilingual model with valid language should succeed."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-multilingual",
            "input": "Bonjour le monde",
            "voice": "default",
            "language": "fr",
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["model_name"] == "chatterbox-multilingual"
    assert call_kwargs.kwargs["language"] == "fr"


async def test_speech_multilingual_missing_language(client: AsyncClient, mock_engine: MagicMock):
    """Multilingual model without language should return 400."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    with patch(
        "chatterbox_service.routes.speech.validate_language",
        side_effect=ValueError("'language' is required for chatterbox-multilingual."),
    ):
        resp = await client.post(
            "/v1/audio/speech",
            json={
                "model": "chatterbox-multilingual",
                "input": "Missing language",
                "voice": "default",
            },
        )
    assert resp.status_code == 400
    assert "language" in resp.json()["detail"].lower()


async def test_speech_multilingual_invalid_language(client: AsyncClient, mock_engine: MagicMock):
    """Multilingual model with unsupported language should return 400."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    with patch(
        "chatterbox_service.routes.speech.validate_language",
        side_effect=ValueError("Unsupported language 'xx'."),
    ):
        resp = await client.post(
            "/v1/audio/speech",
            json={
                "model": "chatterbox-multilingual",
                "input": "Invalid language",
                "voice": "default",
                "language": "xx",
            },
        )
    assert resp.status_code == 400
    assert "xx" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Speech — extended parameters
# ---------------------------------------------------------------------------


async def test_speech_with_seed(client: AsyncClient, mock_engine: MagicMock):
    """Seed parameter should be passed through to engine."""
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Seed test",
            "voice": "default",
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["seed"] == 42


async def test_speech_with_sampling_params(client: AsyncClient, mock_engine: MagicMock):
    """All sampling parameters should be passed through to engine."""
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Sampling params test",
            "voice": "default",
            "temperature": 0.5,
            "repetition_penalty": 1.5,
            "top_p": 0.9,
            "top_k": 500,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["temperature"] == 0.5
    assert call_kwargs.kwargs["repetition_penalty"] == 1.5
    assert call_kwargs.kwargs["top_p"] == 0.9
    assert call_kwargs.kwargs["top_k"] == 500


# ---------------------------------------------------------------------------
# Speech — error cases
# ---------------------------------------------------------------------------


async def test_speech_unknown_model(client: AsyncClient, mock_engine: MagicMock):
    mock_engine.ensure_model.side_effect = ValueError("Unknown model 'nonexistent'.")
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


async def test_speech_generation_failure(client: AsyncClient, mock_engine: MagicMock):
    """Engine exception should result in 500."""
    mock_engine.generate.side_effect = RuntimeError("GPU out of memory")
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Fail test",
            "voice": "default",
        },
    )
    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Speech — default parameter values
# ---------------------------------------------------------------------------


async def test_speech_default_params(client: AsyncClient, mock_engine: MagicMock):
    """When only required fields are sent, defaults should be passed to engine."""
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Defaults test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["speed"] == 1.0
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["top_p"] == 1.0
    assert call_kwargs["min_p"] == 0.05
    assert call_kwargs["top_k"] == 1000
    assert call_kwargs["exaggeration"] == 0.5
    assert call_kwargs["cfg_weight"] == 0.5
    assert call_kwargs["seed"] is None
    assert call_kwargs["language"] is None

"""Tests for the WhisperX STT API endpoints.

These tests mock the engine so they can run without a GPU or model weights.
"""

from __future__ import annotations

import io
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": " Hello world.",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 1.0, "score": 0.95},
            {"word": "world.", "start": 1.1, "end": 2.5, "score": 0.90},
        ],
    },
    {
        "start": 3.0,
        "end": 5.0,
        "text": " This is a test.",
        "words": [
            {"word": "This", "start": 3.0, "end": 3.3, "score": 0.88},
            {"word": "is", "start": 3.4, "end": 3.5, "score": 0.92},
            {"word": "a", "start": 3.6, "end": 3.7, "score": 0.99},
            {"word": "test.", "start": 3.8, "end": 5.0, "score": 0.91},
        ],
    },
]

MOCK_RESULT = {
    "segments": MOCK_SEGMENTS,
    "language": "en",
    "text": "Hello world. This is a test.",
    "duration": 5.0,
}


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    num_samples = int(sample_rate * duration_s)
    # PCM 16-bit mono
    data = b"\x00\x00" * num_samples
    data_size = len(data)
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_engine():
    """Patch the engine singleton so no real model is loaded."""
    with patch("whisperx_service.engine.engine") as eng:
        eng.is_loaded = True
        eng.loaded_model_name = "large-v3"
        eng.list_available_models.return_value = ["whisper-1", "large-v3"]
        eng.ensure_model.return_value = None
        eng.transcribe.return_value = MOCK_RESULT
        # Also patch the engine used in routes
        with (
            patch("whisperx_service.routes.transcriptions.engine", eng),
            patch("whisperx_service.routes.translations.engine", eng),
            patch("whisperx_service.routes.health.engine", eng),
            patch("whisperx_service.routes.models.engine", eng),
        ):
            yield eng


@pytest.fixture()
def app(mock_engine):
    """Create a test app that skips the real lifespan (model loading)."""
    from fastapi import FastAPI

    from whisperx_service.routes import health, models, transcriptions, translations

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(transcriptions.router)
    test_app.include_router(translations.router)
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
    assert len(data["data"]) == 2
    model_ids = [m["id"] for m in data["data"]]
    assert "whisper-1" in model_ids


# ---------------------------------------------------------------------------
# Transcriptions
# ---------------------------------------------------------------------------


async def test_transcription_json(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert data["text"] == "Hello world. This is a test."
    mock_engine.ensure_model.assert_called_once_with("whisper-1")
    mock_engine.transcribe.assert_called_once()


async def test_transcription_text(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Hello world. This is a test."


async def test_transcription_verbose_json(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "transcribe"
    assert data["language"] == "en"
    assert data["duration"] == 5.0
    assert len(data["segments"]) == 2
    assert len(data["words"]) > 0
    assert data["words"][0]["word"] == "Hello"


async def test_transcription_srt(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "srt"},
    )
    assert resp.status_code == 200
    assert "1\n" in resp.text
    assert "-->" in resp.text


async def test_transcription_vtt(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "vtt"},
    )
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")
    assert "-->" in resp.text


async def test_transcription_with_language(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "language": "de"},
    )
    assert resp.status_code == 200
    # Verify language was passed through
    call_kwargs = mock_engine.transcribe.call_args
    assert call_kwargs.kwargs["language"] == "de"


async def test_transcription_invalid_format(client: AsyncClient):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "invalid"},
    )
    assert resp.status_code == 400


async def test_transcription_empty_file(client: AsyncClient):
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------


async def test_translation_json(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    # Verify task was set to translate
    call_kwargs = mock_engine.transcribe.call_args
    assert call_kwargs.kwargs["task"] == "translate"


async def test_translation_empty_file(client: AsyncClient):
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400

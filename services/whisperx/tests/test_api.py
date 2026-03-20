"""Tests for the WhisperX STT backend (Starlette).

These tests mock the engine so they can run without a GPU or model weights.
"""

from __future__ import annotations

import io
import struct
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
        eng.list_available_models.return_value = [
            "whisper-1", "base", "base.en", "large", "large-v1", "large-v2",
            "large-v3", "medium", "medium.en", "small", "small.en", "tiny",
            "tiny.en", "turbo",
        ]
        eng.ensure_model.return_value = None
        eng.transcribe.return_value = MOCK_RESULT
        eng.touch.return_value = None
        # Also patch the engine used in app.py
        with patch("whisperx_service.app.engine", eng):
            yield eng


@pytest.fixture()
def app(mock_engine):
    """Create a test app that skips the real lifespan (model loading)."""
    from starlette.applications import Starlette
    from starlette.routing import Route

    from whisperx_service.app import health, list_models, transcribe, translate

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    return Starlette(
        routes=[
            Route("/transcribe", transcribe, methods=["POST"]),
            Route("/translate", translate, methods=["POST"]),
            Route("/models", list_models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ],
        lifespan=noop_lifespan,
    )


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
    assert data["loaded_model"] == "large-v3"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


async def test_list_models(client: AsyncClient):
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 2
    model_ids = [m["id"] for m in data["data"]]
    assert "whisper-1" in model_ids
    assert "large-v3" in model_ids


async def test_list_models_capabilities(client: AsyncClient):
    """Models endpoint returns capabilities for each model."""
    resp = await client.get("/models")
    data = resp.json()
    for model in data["data"]:
        assert "capabilities" in model
        assert "transcription" in model["capabilities"]
        assert "translation" in model["capabilities"]


async def test_list_models_loaded_status(client: AsyncClient):
    """Models endpoint shows loaded status based on current model."""
    resp = await client.get("/models")
    data = resp.json()
    by_id = {m["id"]: m for m in data["data"]}
    # whisper-1 alias should show as loaded when any model is loaded
    assert by_id["whisper-1"]["loaded"] is True
    # The actual loaded model should show as loaded
    assert by_id["large-v3"]["loaded"] is True
    # Other models should not be loaded
    assert by_id["tiny"]["loaded"] is False


async def test_list_models_all_known_sizes(client: AsyncClient):
    """All known whisper model sizes are listed."""
    resp = await client.get("/models")
    data = resp.json()
    model_ids = {m["id"] for m in data["data"]}
    assert "tiny" in model_ids
    assert "large-v3" in model_ids
    assert "turbo" in model_ids


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


async def test_transcribe_returns_json(client: AsyncClient, mock_engine: MagicMock):
    """Transcription endpoint returns raw segments JSON."""
    wav = _make_wav_bytes()
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world. This is a test."
    assert data["language"] == "en"
    assert data["duration"] == 5.0
    assert len(data["segments"]) == 2
    mock_engine.ensure_model.assert_called_once_with("whisper-1")
    mock_engine.transcribe.assert_called_once()


async def test_transcribe_with_language(client: AsyncClient, mock_engine: MagicMock):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "language": "de"},
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.transcribe.call_args
    assert call_kwargs.kwargs["language"] == "de"


async def test_transcribe_with_word_timestamps(client: AsyncClient, mock_engine: MagicMock):
    """word_timestamps=true should be passed to engine."""
    wav = _make_wav_bytes()
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "word_timestamps": "true"},
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.transcribe.call_args
    assert call_kwargs.kwargs["word_timestamps"] is True


async def test_transcribe_with_diarize(client: AsyncClient, mock_engine: MagicMock):
    """diarize=true should be passed to engine and force word_timestamps."""
    from whisperx_service.config import settings

    original = settings.enable_diarization
    settings.enable_diarization = True
    try:
        wav = _make_wav_bytes()
        resp = await client.post(
            "/transcribe",
            files={"file": ("test.wav", wav, "audio/wav")},
            data={"model": "whisper-1", "diarize": "true"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_engine.transcribe.call_args
        assert call_kwargs.kwargs["diarize"] is True
        # Diarization forces word timestamps on
        assert call_kwargs.kwargs["word_timestamps"] is True
    finally:
        settings.enable_diarization = original


async def test_transcribe_diarize_without_config(client: AsyncClient, mock_engine: MagicMock):
    """diarize=true is ignored when enable_diarization is false in config."""
    from whisperx_service.config import settings

    original = settings.enable_diarization
    settings.enable_diarization = False
    try:
        wav = _make_wav_bytes()
        resp = await client.post(
            "/transcribe",
            files={"file": ("test.wav", wav, "audio/wav")},
            data={"model": "whisper-1", "diarize": "true"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_engine.transcribe.call_args
        assert call_kwargs.kwargs["diarize"] is False
    finally:
        settings.enable_diarization = original


async def test_transcribe_empty_file(client: AsyncClient):
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400


async def test_transcribe_missing_model(client: AsyncClient):
    wav = _make_wav_bytes()
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 400


async def test_transcribe_failure(client: AsyncClient, mock_engine: MagicMock):
    """Engine exception should result in 500."""
    mock_engine.transcribe.side_effect = RuntimeError("GPU error")
    wav = _make_wav_bytes()
    resp = await client.post(
        "/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


async def test_translate_returns_json(client: AsyncClient, mock_engine: MagicMock):
    """Translation endpoint returns raw segments JSON."""
    wav = _make_wav_bytes()
    resp = await client.post(
        "/translate",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "segments" in data
    # Verify task was set to translate
    call_kwargs = mock_engine.transcribe.call_args
    assert call_kwargs.kwargs["task"] == "translate"


async def test_translate_empty_file(client: AsyncClient):
    resp = await client.post(
        "/translate",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400


async def test_translate_failure(client: AsyncClient, mock_engine: MagicMock):
    """Engine exception should result in 500."""
    mock_engine.transcribe.side_effect = RuntimeError("GPU error")
    wav = _make_wav_bytes()
    resp = await client.post(
        "/translate",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()

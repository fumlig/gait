"""Tests for the API gateway.

These tests mock the client instances to avoid needing real backend services.
Clients are attached directly to ``app.state``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from gateway_service.models import (
    ModelObject,
    RawSegment,
    TranscriptionResult,
    Voice,
    WordTimestamp,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway_service.models import SpeechRequest

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

CHATTERBOX_MODELS = [
    ModelObject(id="chatterbox-turbo", owned_by="resemble-ai"),
]

WHISPERX_MODELS = [
    ModelObject(id="whisper-1", owned_by="whisperx"),
    ModelObject(id="large-v3", owned_by="whisperx"),
]

ALL_MODELS = CHATTERBOX_MODELS + WHISPERX_MODELS

# Minimal valid WAV (44-byte header + 2 bytes of silence)
_WAV_HEADER = (
    b"RIFF"
    b"\x2e\x00\x00\x00"  # file size - 8
    b"WAVE"
    b"fmt "
    b"\x10\x00\x00\x00"  # chunk size
    b"\x01\x00"  # PCM
    b"\x01\x00"  # mono
    b"\x80\x3e\x00\x00"  # 16000 Hz
    b"\x00\x7d\x00\x00"  # byte rate
    b"\x02\x00"  # block align
    b"\x10\x00"  # 16 bit
    b"data"
    b"\x02\x00\x00\x00"  # data size
    b"\x00\x00"  # one sample of silence
)

MOCK_TRANSCRIPTION_RESULT = TranscriptionResult(
    text="Hello world. This is a test.",
    language="en",
    duration=5.0,
    segments=[
        RawSegment(
            start=0.0,
            end=2.5,
            text="Hello world.",
            words=[
                WordTimestamp(word="Hello", start=0.0, end=1.0, score=0.95),
                WordTimestamp(word="world.", start=1.1, end=2.5, score=0.90),
            ],
        ),
        RawSegment(
            start=3.0,
            end=5.0,
            text="This is a test.",
            words=[
                WordTimestamp(word="This", start=3.0, end=3.3, score=0.88),
                WordTimestamp(word="is", start=3.4, end=3.5, score=0.92),
                WordTimestamp(word="a", start=3.6, end=3.7, score=0.99),
                WordTimestamp(word="test.", start=3.8, end=5.0, score=0.91),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Mock client factories
# ---------------------------------------------------------------------------


def _make_speech_client(
    *,
    synthesize_result: tuple[bytes, str] = (_WAV_HEADER, "audio/wav"),
    synthesize_error: Exception | None = None,
) -> AsyncMock:
    """Create a mock SpeechClient."""
    client = AsyncMock()
    client.health.return_value = {"status": "healthy"}
    client.list_models.return_value = CHATTERBOX_MODELS
    if synthesize_error:
        client.synthesize.side_effect = synthesize_error
    else:
        client.synthesize.return_value = synthesize_result
    return client


def _make_transcription_client(
    *,
    transcribe_result: TranscriptionResult | None = None,
    translate_result: TranscriptionResult | None = None,
    transcribe_error: Exception | None = None,
) -> AsyncMock:
    """Create a mock TranscriptionClient."""
    client = AsyncMock()
    client.health.return_value = {"status": "healthy"}
    client.list_models.return_value = WHISPERX_MODELS

    if transcribe_error:
        client.transcribe.side_effect = transcribe_error
    else:
        client.transcribe.return_value = transcribe_result or MOCK_TRANSCRIPTION_RESULT

    client.translate.return_value = translate_result or TranscriptionResult(
        text="Translated text",
        language="en",
        duration=3.0,
        segments=[
            RawSegment(start=0.0, end=3.0, text="Translated text"),
        ],
    )
    return client


def _make_voice_client(
    *,
    voices: list[Voice] | None = None,
) -> AsyncMock:
    """Create a mock VoiceClient."""
    client = AsyncMock()
    client.health.return_value = {"status": "healthy"}

    default_voices = voices or [
        Voice(voice_id="default", name="default"),
        Voice(voice_id="narrator", name="narrator"),
    ]
    client.list_voices.return_value = default_voices
    client.get_voice.return_value = default_voices[0]
    client.create_voice.return_value = Voice(voice_id="newvoice", name="newvoice")
    client.delete_voice.return_value = {"deleted": True, "voice_id": "default"}
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def speech_client():
    return _make_speech_client()


@pytest.fixture()
def transcription_client():
    return _make_transcription_client()


@pytest.fixture()
def voice_client():
    return _make_voice_client()


@pytest.fixture()
def app(speech_client, transcription_client, voice_client):
    """Create a test app with mock clients attached to state."""
    from fastapi import FastAPI

    from gateway_service.routes import health, models
    from gateway_service.routes.audio import router as audio_router

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(audio_router)
    test_app.include_router(models.router)
    test_app.include_router(health.router)

    test_app.state.speech_client = speech_client
    test_app.state.transcription_client = transcription_client
    test_app.state.voice_client = voice_client
    test_app.state.models = ALL_MODELS

    return test_app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_all_healthy(client: AsyncClient):
    """Gateway reports 'ok' when all backends are healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backends"]["speech"] == "healthy"
    assert data["backends"]["transcription"] == "healthy"
    assert data["backends"]["voice"] == "healthy"


async def test_health_backend_down(client: AsyncClient, speech_client: AsyncMock):
    """Gateway reports 'degraded' when a backend is unreachable."""
    speech_client.health.side_effect = ConnectionError("Connection refused")
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["backends"]["speech"] == "unreachable"


# ---------------------------------------------------------------------------
# Models (cached)
# ---------------------------------------------------------------------------


async def test_list_models_merged(client: AsyncClient):
    """Gateway returns the cached merged model list."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = {m["id"] for m in data["data"]}
    assert "chatterbox-turbo" in model_ids
    assert "whisper-1" in model_ids
    assert "large-v3" in model_ids


async def test_list_models_empty_when_no_models():
    """Gateway returns an empty list when no models were discovered."""
    from fastapi import FastAPI

    from gateway_service.routes import models

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(models.router)
    # No models on state
    test_app.state.models = []

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == []


# ---------------------------------------------------------------------------
# Audio: Speech
# ---------------------------------------------------------------------------


async def test_speech_wav(client: AsyncClient, speech_client: AsyncMock):
    """Speech requests with WAV format return WAV audio directly."""
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello",
            "voice": "default",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
    speech_client.synthesize.assert_called_once()


async def test_speech_default_format(client: AsyncClient, speech_client: AsyncMock):
    """Speech requests call the speech client (default format is mp3)."""
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "chatterbox-turbo", "input": "Hello", "voice": "default"},
    )
    assert resp.status_code == 200
    speech_client.synthesize.assert_called_once()

    # Verify the SpeechRequest was passed correctly
    call_args = speech_client.synthesize.call_args
    req: SpeechRequest = call_args.args[0]
    assert req.model == "chatterbox-turbo"
    assert req.input == "Hello"
    assert req.voice == "default"


async def test_speech_backend_unavailable(client: AsyncClient, speech_client: AsyncMock):
    """Gateway returns 502 when the speech backend raises."""
    speech_client.synthesize.side_effect = ConnectionError("Connection refused")
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "chatterbox-turbo", "input": "Hello", "voice": "default"},
    )
    assert resp.status_code == 502


async def test_speech_validation_error(client: AsyncClient):
    """Gateway returns 422 for invalid request body."""
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "chatterbox-turbo"},  # missing required fields
    )
    assert resp.status_code == 422


async def test_speech_unsupported_format(client: AsyncClient, speech_client: AsyncMock):
    """Gateway returns 400 for unsupported audio formats."""
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "chatterbox-turbo",
            "input": "Hello",
            "voice": "default",
            "response_format": "opus",
        },
    )
    assert resp.status_code == 400
    assert "not currently supported" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Audio: Transcriptions
# ---------------------------------------------------------------------------


async def test_transcription_json(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with json format returns {text: ...}."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world. This is a test."
    transcription_client.transcribe.assert_called_once()

    # Verify kwargs
    call_kwargs = transcription_client.transcribe.call_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["filename"] == "test.wav"
    assert call_kwargs["file"] == b"fake-wav"
    assert call_kwargs["word_timestamps"] is False


async def test_transcription_text(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with text format returns plain text."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Hello world. This is a test."


async def test_transcription_verbose_json(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with verbose_json returns full segments and words."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
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

    # verbose_json should request word timestamps
    call_kwargs = transcription_client.transcribe.call_args.kwargs
    assert call_kwargs["word_timestamps"] is True


async def test_transcription_srt(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with srt format returns SRT subtitles."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "srt"},
    )
    assert resp.status_code == 200
    assert "1\n" in resp.text
    assert "-->" in resp.text
    assert "Hello world." in resp.text


async def test_transcription_vtt(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with vtt format returns WebVTT subtitles."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "vtt"},
    )
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")
    assert "-->" in resp.text


async def test_transcription_invalid_format(client: AsyncClient):
    """Gateway returns 400 for invalid response_format."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "invalid"},
    )
    assert resp.status_code == 400
    assert "Invalid response_format" in resp.json()["detail"]


async def test_transcription_empty_file(client: AsyncClient):
    """Gateway rejects empty audio files."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400


async def test_transcription_backend_unavailable(
    client: AsyncClient, transcription_client: AsyncMock
):
    """Gateway returns 502 when the transcription backend raises."""
    transcription_client.transcribe.side_effect = ConnectionError("down")
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Audio: Translations
# ---------------------------------------------------------------------------


async def test_translation_json(client: AsyncClient, transcription_client: AsyncMock):
    """Translation requests call the transcription client's translate method."""
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Translated text"
    transcription_client.translate.assert_called_once()


async def test_translation_empty_file(client: AsyncClient):
    """Gateway rejects empty audio files for translation."""
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400


async def test_translation_invalid_format(client: AsyncClient):
    """Gateway returns 400 for invalid response_format on translations."""
    resp = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "invalid"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Audio: Voices
# ---------------------------------------------------------------------------


async def test_list_voices(client: AsyncClient, voice_client: AsyncMock):
    """GET /v1/audio/voices returns a wrapped list."""
    resp = await client.get("/v1/audio/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    names = {v["name"] for v in data["data"]}
    assert names == {"default", "narrator"}
    voice_client.list_voices.assert_called_once()


async def test_get_voice(client: AsyncClient, voice_client: AsyncMock):
    """GET /v1/audio/voices/{name} returns a voice object."""
    resp = await client.get("/v1/audio/voices/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "default"
    assert data["name"] == "default"
    voice_client.get_voice.assert_called_once_with("default")


async def test_get_voice_not_found(client: AsyncClient, voice_client: AsyncMock):
    """GET /v1/audio/voices/{name} returns 404 when client raises."""
    from fastapi import HTTPException

    voice_client.get_voice.side_effect = HTTPException(status_code=404, detail="Not found")
    resp = await client.get("/v1/audio/voices/nonexistent")
    assert resp.status_code == 404


async def test_create_voice(client: AsyncClient, voice_client: AsyncMock):
    """POST /v1/audio/voices creates a new voice."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "newvoice"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["voice_id"] == "newvoice"
    assert data["name"] == "newvoice"
    voice_client.create_voice.assert_called_once_with("newvoice", _WAV_HEADER)


async def test_create_voice_empty_file(client: AsyncClient):
    """POST /v1/audio/voices rejects empty uploads."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "empty"},
        files={"file": ("sample.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


async def test_create_voice_conflict(client: AsyncClient, voice_client: AsyncMock):
    """POST /v1/audio/voices returns 409 when client raises conflict."""
    from fastapi import HTTPException

    voice_client.create_voice.side_effect = HTTPException(status_code=409, detail="Already exists")
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "existing"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 409


async def test_delete_voice(client: AsyncClient, voice_client: AsyncMock):
    """DELETE /v1/audio/voices/{name} deletes a voice."""
    resp = await client.delete("/v1/audio/voices/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    voice_client.delete_voice.assert_called_once_with("default")


async def test_delete_voice_not_found(client: AsyncClient, voice_client: AsyncMock):
    """DELETE /v1/audio/voices/{name} returns 404 when client raises."""
    from fastapi import HTTPException

    voice_client.delete_voice.side_effect = HTTPException(status_code=404, detail="Not found")
    resp = await client.delete("/v1/audio/voices/nonexistent")
    assert resp.status_code == 404


async def test_voice_client_unavailable():
    """Gateway returns 503 when no voice client is configured."""
    from fastapi import FastAPI

    from gateway_service.routes.audio import router as audio_router

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(audio_router)
    # No voice client on state

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/audio/voices")
    assert resp.status_code == 503

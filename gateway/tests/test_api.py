from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from gateway_service.models import (
    ModelObject,
    RawSegment,
    TranscriptionResult,
    WordTimestamp,
)
from gateway_service.providers.voice import VoiceClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway_service.models import SpeechRequest

CHATTERBOX_MODELS = [
    ModelObject(
        id="chatterbox-turbo",
        owned_by="resemble-ai",
        capabilities=["speech"],
        loaded=True,
    ),
]

WHISPERX_MODELS = [
    ModelObject(
        id="whisper-1",
        owned_by="whisperx",
        capabilities=["transcription", "translation"],
        loaded=True,
    ),
    ModelObject(
        id="large-v3",
        owned_by="whisperx",
        capabilities=["transcription", "translation"],
        loaded=True,
    ),
]

LLAMACPP_MODELS = [
    ModelObject(
        id="my-model",
        owned_by="llamacpp",
        capabilities=["chat", "completions", "embeddings"],
        loaded=True,
    ),
]

ALL_MODELS = CHATTERBOX_MODELS + WHISPERX_MODELS + LLAMACPP_MODELS

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


def _make_mock_stream():
    from starlette.responses import StreamingResponse

    async def _gen():
        yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _make_speech_client(
    *,
    synthesize_result: tuple[bytes, str] = (_WAV_HEADER, "audio/wav"),
    synthesize_error: Exception | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.name = "chatterbox"
    client.base_url = "http://chatterbox:8000"
    client.check_health.return_value = {"status": "healthy"}
    client.fetch_models.return_value = CHATTERBOX_MODELS
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
    client = AsyncMock()
    client.name = "whisperx"
    client.base_url = "http://whisperx:8000"
    client.check_health.return_value = {"status": "healthy"}
    client.fetch_models.return_value = WHISPERX_MODELS

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


MOCK_CHAT_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

MOCK_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
}

MOCK_PARALLEL_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool-2",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    },
                    {
                        "id": "call_def456",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 60, "completion_tokens": 50, "total_tokens": 110},
}

MOCK_COMPLETION = {
    "id": "cmpl-123",
    "object": "text_completion",
    "created": 1700000000,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "text": "Hello world",
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

MOCK_RESPONSE = {
    "id": "resp-123",
    "object": "response",
    "created_at": 1700000000,
    "model": "my-model",
    "output": [
        {
            "type": "message",
            "id": "msg-123",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello! How can I help?"}],
        }
    ],
    "status": "completed",
    "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
}

MOCK_RESPONSE_WITH_TOOL_CALLS = {
    "id": "resp-tool-1",
    "object": "response",
    "created_at": 1700000000,
    "model": "my-model",
    "output": [
        {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
        }
    ],
    "status": "completed",
    "usage": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
}

MOCK_EMBEDDINGS = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3],
        }
    ],
    "model": "my-model",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


def _make_chat_client() -> AsyncMock:
    client = AsyncMock()
    client.name = "llamacpp"
    client.base_url = "http://llamacpp:8000"
    client.check_health.return_value = {"status": "healthy"}
    client.fetch_models.return_value = LLAMACPP_MODELS

    # ChatCompletions
    client.chat_completions.return_value = MOCK_CHAT_COMPLETION
    client.chat_completions_stream.return_value = _make_mock_stream()

    # Completions
    client.completions.return_value = MOCK_COMPLETION
    client.completions_stream.return_value = _make_mock_stream()

    # Responses
    client.create_response.return_value = MOCK_RESPONSE
    client.create_response_stream.return_value = _make_mock_stream()

    # Embeddings
    client.embeddings.return_value = MOCK_EMBEDDINGS

    return client


@pytest.fixture()
def speech_client():
    return _make_speech_client()


@pytest.fixture()
def transcription_client():
    return _make_transcription_client()


@pytest.fixture()
def voice_client(tmp_path):
    return VoiceClient(voices_dir=tmp_path)


@pytest.fixture()
def chat_client():
    return _make_chat_client()


@pytest.fixture()
def app(speech_client, transcription_client, voice_client, chat_client):
    """Create a test app with mock clients wired to protocol slots."""
    from fastapi import FastAPI

    from gateway_service.routes import completions, embeddings, health, models, responses
    from gateway_service.routes.audio import router as audio_router
    from gateway_service.routes.chat import router as chat_router

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(chat_router)
    test_app.include_router(completions.router)
    test_app.include_router(responses.router)
    test_app.include_router(embeddings.router)
    test_app.include_router(audio_router)
    test_app.include_router(models.router)
    test_app.include_router(health.router)

    # Wire protocol slots (mirrors what the real lifespan does)
    test_app.state.chat_completions = chat_client
    test_app.state.completions = chat_client
    test_app.state.responses = chat_client
    test_app.state.embeddings = chat_client
    test_app.state.audio_speech = speech_client
    test_app.state.audio_transcriptions = transcription_client
    test_app.state.audio_translations = transcription_client
    test_app.state.audio_voices = voice_client

    # Provider list for health and model discovery
    test_app.state.providers = [speech_client, transcription_client, chat_client]

    test_app.state.models = ALL_MODELS
    test_app.state.models_fetched_at = time.monotonic()

    return test_app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===================================================================
# Health
# ===================================================================


async def test_health_all_healthy(client: AsyncClient):
    """Gateway reports 'ok' when all backends are healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backends"]["chatterbox"] == "healthy"
    assert data["backends"]["whisperx"] == "healthy"
    assert data["backends"]["llamacpp"] == "healthy"
    # Voice is local — not listed as a remote backend
    assert "voice" not in data["backends"]


async def test_health_backend_down(client: AsyncClient, speech_client: AsyncMock):
    """Gateway reports 'degraded' when a backend is unreachable."""
    speech_client.check_health.side_effect = ConnectionError("Connection refused")
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["backends"]["chatterbox"] == "unreachable"


# ===================================================================
# Models
# ===================================================================


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
    assert "my-model" in model_ids


async def test_list_models_capabilities(client: AsyncClient):
    """Gateway models include capabilities."""
    resp = await client.get("/v1/models")
    data = resp.json()
    by_id = {m["id"]: m for m in data["data"]}
    assert "speech" in by_id["chatterbox-turbo"]["capabilities"]
    assert "transcription" in by_id["whisper-1"]["capabilities"]
    assert "chat" in by_id["my-model"]["capabilities"]
    assert "embeddings" in by_id["my-model"]["capabilities"]


async def test_list_models_loaded_status(client: AsyncClient):
    """Gateway models include loaded status."""
    resp = await client.get("/v1/models")
    data = resp.json()
    for m in data["data"]:
        assert "loaded" in m


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
    test_app.state.models_fetched_at = time.monotonic()

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == []


# ===================================================================
# Audio — Speech
# ===================================================================


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


# ===================================================================
# Audio — Transcriptions
# ===================================================================


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


async def test_transcription_diarize(client: AsyncClient, transcription_client: AsyncMock):
    """Transcription with diarize=true passes through to the backend."""
    resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-wav", "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json", "diarize": "true"},
    )
    assert resp.status_code == 200
    call_kwargs = transcription_client.transcribe.call_args.kwargs
    assert call_kwargs["diarize"] is True
    # Diarization forces word_timestamps
    assert call_kwargs["word_timestamps"] is True


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


# ===================================================================
# Audio — Translations
# ===================================================================


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


# ===================================================================
# Audio — Voices
# ===================================================================


async def test_list_voices_empty(client: AsyncClient):
    """GET /v1/audio/voices returns 'default' even with empty dir."""
    resp = await client.get("/v1/audio/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["name"] == "default"


async def test_list_voices_with_files(client: AsyncClient, tmp_path):
    """GET /v1/audio/voices includes disk voices alongside 'default'."""
    (tmp_path / "alice.wav").write_bytes(_WAV_HEADER)
    (tmp_path / "bob.wav").write_bytes(_WAV_HEADER)
    resp = await client.get("/v1/audio/voices")
    assert resp.status_code == 200
    names = [v["name"] for v in resp.json()["data"]]
    assert names == ["default", "alice", "bob"]


async def test_get_voice(client: AsyncClient):
    """GET /v1/audio/voices/default returns the built-in default voice."""
    resp = await client.get("/v1/audio/voices/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "default"
    assert data["name"] == "default"


async def test_get_voice_not_found(client: AsyncClient):
    """GET /v1/audio/voices/{name} returns 404 for unknown voices."""
    resp = await client.get("/v1/audio/voices/nonexistent")
    assert resp.status_code == 404


async def test_create_voice(client: AsyncClient, tmp_path):
    """POST /v1/audio/voices creates a new voice on disk."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "newvoice"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["voice_id"] == "newvoice"
    assert data["name"] == "newvoice"
    assert (tmp_path / "newvoice.wav").exists()


async def test_create_voice_empty_file(client: AsyncClient):
    """POST /v1/audio/voices rejects empty uploads."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "empty"},
        files={"file": ("sample.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


async def test_create_voice_invalid_name(client: AsyncClient):
    """POST /v1/audio/voices rejects invalid voice names."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "bad name!"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "alphanumeric" in resp.json()["detail"]


async def test_create_voice_duplicate(client: AsyncClient, tmp_path):
    """POST /v1/audio/voices returns 409 for existing voices."""
    (tmp_path / "existing.wav").write_bytes(_WAV_HEADER)
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "existing"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 409


async def test_create_voice_not_wav(client: AsyncClient):
    """POST /v1/audio/voices rejects non-WAV files."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "bad"},
        files={"file": ("sample.mp3", b"\xff\xfb\x90\x00" + b"\x00" * 100, "audio/mpeg")},
    )
    assert resp.status_code == 400
    assert "not a valid WAV" in resp.json()["detail"]


async def test_create_voice_too_small(client: AsyncClient):
    """POST /v1/audio/voices rejects files too small to be WAV."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "tiny"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"].lower()


async def test_create_voice_default_rejected(client: AsyncClient):
    """POST /v1/audio/voices rejects 'default' (reserved built-in name)."""
    resp = await client.post(
        "/v1/audio/voices",
        data={"name": "default"},
        files={"file": ("sample.wav", _WAV_HEADER, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "built-in" in resp.json()["detail"]


async def test_delete_voice(client: AsyncClient, tmp_path):
    """DELETE /v1/audio/voices/{name} removes the voice file."""
    voice_file = tmp_path / "todelete.wav"
    voice_file.write_bytes(_WAV_HEADER)
    resp = await client.delete("/v1/audio/voices/todelete")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["voice_id"] == "todelete"
    assert not voice_file.exists()


async def test_delete_voice_not_found(client: AsyncClient):
    """DELETE /v1/audio/voices/{name} returns 404 for unknown voices."""
    resp = await client.delete("/v1/audio/voices/nonexistent")
    assert resp.status_code == 404


async def test_delete_default_voice_rejected(client: AsyncClient):
    """DELETE /v1/audio/voices/default is rejected (built-in)."""
    resp = await client.delete("/v1/audio/voices/default")
    assert resp.status_code == 400
    assert "built-in" in resp.json()["detail"]


async def test_voice_client_unavailable():
    """Gateway returns 503 when no voice client is configured."""
    from fastapi import FastAPI

    from gateway_service.routes.audio import router as audio_router

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(audio_router)
    # No audio_voices on state

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/v1/audio/voices")
    assert resp.status_code == 503


# ===================================================================
# Chat Completions — basic
# ===================================================================


async def test_chat_completions(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/chat/completions returns a chat completion response."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
    chat_client.chat_completions.assert_called_once_with(
        {"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]},
    )


async def test_chat_completions_stream(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/chat/completions with stream=true returns an SSE stream."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data:" in resp.text
    chat_client.chat_completions_stream.assert_called_once_with(
        {"model": "my-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )


async def test_chat_completions_backend_unavailable(
    client: AsyncClient, chat_client: AsyncMock
):
    """Gateway returns 502 when the chat backend raises."""
    chat_client.chat_completions.side_effect = ConnectionError("Connection refused")
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 502


# ===================================================================
# Chat Completions — tool calling
# ===================================================================


async def test_chat_completions_with_tools(client: AsyncClient, chat_client: AsyncMock):
    """Chat completion with tools defined returns tool_calls in assistant message."""
    chat_client.chat_completions.return_value = MOCK_TOOL_CALL_RESPONSE

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"

    tool_calls = data["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_abc123"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"

    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["location"] == "San Francisco"
    assert args["unit"] == "celsius"

    # Verify the request was forwarded with tools
    call_args = chat_client.chat_completions.call_args[0][0]
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["function"]["name"] == "get_weather"
    assert call_args["tool_choice"] == "auto"


async def test_chat_completions_tool_results(client: AsyncClient, chat_client: AsyncMock):
    """Messages can include tool results for multi-turn tool calling."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"temperature": 72, "unit": "fahrenheit"}',
                    "tool_call_id": "call_abc123",
                },
            ],
        },
    )

    assert resp.status_code == 200

    # Verify all messages were forwarded correctly
    call_args = chat_client.chat_completions.call_args[0][0]
    messages = call_args["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    assert len(messages[1]["tool_calls"]) == 1
    assert messages[1]["tool_calls"][0]["id"] == "call_abc123"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_abc123"


async def test_chat_completions_parallel_tool_calls(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Model can issue multiple tool calls in a single response."""
    chat_client.chat_completions.return_value = MOCK_PARALLEL_TOOL_CALL_RESPONSE

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "Weather in SF and Tokyo?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            "parallel_tool_calls": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    tool_calls = data["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[1]["function"]["name"] == "get_weather"

    # Verify parallel_tool_calls was forwarded
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["parallel_tool_calls"] is True


async def test_chat_completions_tool_choice_forced(
    client: AsyncClient, chat_client: AsyncMock,
):
    """tool_choice can force a specific function to be called."""
    chat_client.chat_completions.return_value = MOCK_TOOL_CALL_RESPONSE

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["tool_choice"]["type"] == "function"
    assert call_args["tool_choice"]["function"]["name"] == "get_weather"


async def test_chat_completions_tool_choice_none(
    client: AsyncClient, chat_client: AsyncMock,
):
    """tool_choice='none' prevents tool calling."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "none",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["tool_choice"] == "none"


async def test_chat_completions_streaming_tool_calls(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Streaming chat completion with tool calls returns SSE chunks."""
    from starlette.responses import StreamingResponse

    def _sse(delta: dict, *, finish: str | None = None) -> bytes:
        obj = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "my-model",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish},
            ],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()

    async def _gen():
        yield _sse({"role": "assistant", "content": None})
        yield _sse({"tool_calls": [
            {"index": 0, "id": "call_abc", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}},
        ]})
        yield _sse({"tool_calls": [
            {"index": 0, "function": {"arguments": '{"location":"SF"}'}},
        ]})
        yield _sse({}, finish="tool_calls")
        yield b"data: [DONE]\n\n"

    chat_client.chat_completions_stream.return_value = StreamingResponse(
        _gen(), media_type="text/event-stream",
    )

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        },
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    events = []
    for raw in resp.text.strip().split("\n\n"):
        for line in raw.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))

    # First event: role
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    # Second event: tool call start
    assert events[1]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_abc"
    # Third event: tool call arguments
    assert "arguments" in events[2]["choices"][0]["delta"]["tool_calls"][0]["function"]
    # Fourth event: finish
    assert events[3]["choices"][0]["finish_reason"] == "tool_calls"


# ===================================================================
# Chat Completions — response format
# ===================================================================


async def test_chat_completions_json_mode(client: AsyncClient, chat_client: AsyncMock):
    """response_format: json_object is forwarded to the backend."""
    chat_client.chat_completions.return_value = {
        **MOCK_CHAT_COMPLETION,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer": 42}'},
                "finish_reason": "stop",
            }
        ],
    }

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Answer in JSON"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["response_format"]["type"] == "json_object"


async def test_chat_completions_json_schema(client: AsyncClient, chat_client: AsyncMock):
    """response_format: json_schema with a schema is forwarded to the backend."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Give me data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "unit": {"type": "string"},
                        },
                        "required": ["temperature", "unit"],
                    },
                },
            },
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["response_format"]["type"] == "json_schema"
    assert call_args["response_format"]["json_schema"]["name"] == "weather_response"


# ===================================================================
# Chat Completions — sampling & options
# ===================================================================


async def test_chat_completions_sampling_params(
    client: AsyncClient, chat_client: AsyncMock,
):
    """All sampling parameters are forwarded to the backend."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "max_completion_tokens": 200,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "seed": 42,
            "n": 2,
            "stop": ["\n", "END"],
            "logprobs": True,
            "top_logprobs": 5,
            "user": "test-user",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["temperature"] == 0.7
    assert call_args["top_p"] == 0.9
    assert call_args["max_tokens"] == 100
    assert call_args["max_completion_tokens"] == 200
    assert call_args["presence_penalty"] == 0.5
    assert call_args["frequency_penalty"] == 0.3
    assert call_args["seed"] == 42
    assert call_args["n"] == 2
    assert call_args["stop"] == ["\n", "END"]
    assert call_args["logprobs"] is True
    assert call_args["top_logprobs"] == 5
    assert call_args["user"] == "test-user"


async def test_chat_completions_stream_options(
    client: AsyncClient, chat_client: AsyncMock,
):
    """stream_options is forwarded when streaming."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions_stream.call_args[0][0]
    assert call_args["stream_options"]["include_usage"] is True


async def test_chat_completions_logit_bias(client: AsyncClient, chat_client: AsyncMock):
    """logit_bias is forwarded to the backend."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "logit_bias": {"50256": -100, "1": 5.0},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["logit_bias"] == {"50256": -100.0, "1": 5.0}


# ===================================================================
# Chat Completions — message types
# ===================================================================


async def test_chat_completions_system_message(
    client: AsyncClient, chat_client: AsyncMock,
):
    """System messages are forwarded correctly."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are a helpful assistant."


async def test_chat_completions_multimodal_content(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Messages with array content (vision/multimodal) pass through."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    content = call_args["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


async def test_chat_completions_message_name(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Messages with 'name' field pass through."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "Hello", "name": "Alice"},
            ],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["messages"][0]["name"] == "Alice"


# ===================================================================
# Chat Completions — extra fields pass through
# ===================================================================


async def test_chat_completions_extra_fields(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Unknown fields in the request body are forwarded to the backend."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "custom_field": "custom_value",
            "another_extension": 42,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args["custom_field"] == "custom_value"
    assert call_args["another_extension"] == 42


# ===================================================================
# Chat Completions — validation
# ===================================================================


async def test_chat_completions_missing_model(client: AsyncClient):
    """Gateway returns 422 when 'model' is missing."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 422


async def test_chat_completions_missing_messages(client: AsyncClient):
    """Gateway returns 422 when 'messages' is missing."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "my-model"},
    )
    assert resp.status_code == 422


async def test_chat_completions_invalid_message_no_role(client: AsyncClient):
    """Gateway returns 422 when a message has no 'role'."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"content": "Hello"}],
        },
    )
    assert resp.status_code == 422


# ===================================================================
# Chat Completions — audio
# ===================================================================


def _make_wav_24k(num_samples: int = 100) -> bytes:
    """Build a minimal 24 kHz / 16-bit / mono WAV for testing."""
    import struct

    data_size = num_samples * 2
    file_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        24000,
        24000 * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + (b"\x00\x00" * num_samples)


class _MockStreamResponse:
    """Mimics the subset of ``httpx.Response`` used by ``_chat_with_audio``."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        pass


def _sse_chunk(content: str, *, role: bool = False) -> str:
    """Build a single SSE line the way llama.cpp would emit it."""
    delta = {"role": "assistant"} if role else {"content": content}
    obj = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "my-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(obj)}"


_AUDIO_SSE_LINES: list[str] = [
    _sse_chunk("", role=True),
    "",
    _sse_chunk("Hello"),
    "",
    _sse_chunk("!"),
    "",
    _sse_chunk(" How"),
    "",
    _sse_chunk(" are"),
    "",
    _sse_chunk(" you"),
    "",
    _sse_chunk("?"),
    "",
    "data: [DONE]",
]


async def test_chat_audio_stream(
    client: AsyncClient, chat_client: AsyncMock, speech_client: AsyncMock
):
    """Streaming with audio modality interleaves text and audio SSE events."""
    import json as _json

    wav_24k = _make_wav_24k(100)
    speech_client.synthesize.return_value = (wav_24k, "audio/wav")
    chat_client.chat_completions_stream_raw.return_value = _MockStreamResponse(_AUDIO_SSE_LINES)

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "modalities": ["text", "audio"],
            "audio": {"voice": "default", "format": "pcm16"},
        },
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    text_tokens: list[str] = []
    audio_data_chunks: list[str] = []
    audio_transcripts: list[str] = []
    audio_id: str | None = None
    got_expires = False

    for raw_event in resp.text.strip().split("\n\n"):
        for line in raw_event.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            data = _json.loads(payload)
            delta = data["choices"][0]["delta"]
            if "content" in delta:
                text_tokens.append(delta["content"])
            if "audio" in delta:
                audio = delta["audio"]
                if "id" in audio:
                    audio_id = audio["id"]
                if "data" in audio:
                    audio_data_chunks.append(audio["data"])
                if "transcript" in audio:
                    audio_transcripts.append(audio["transcript"])
                if "expires_at" in audio:
                    got_expires = True

    # All text tokens forwarded.
    assert text_tokens == ["Hello", "!", " How", " are", " you", "?"]

    # Audio metadata present.
    assert audio_id is not None
    assert got_expires

    # Two synthesis calls: "Hello!" (sentence boundary) and "How are you?" (flush).
    assert speech_client.synthesize.call_count == 2
    assert len(audio_data_chunks) > 0
    assert audio_transcripts == ["Hello!", "How are you?"]


async def test_chat_audio_requires_stream(client: AsyncClient, chat_client: AsyncMock):
    """Audio modality without stream=true returns 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "default"},
        },
    )
    assert resp.status_code == 400
    assert "stream" in resp.json()["detail"].lower()


async def test_chat_audio_no_speech_backend(client: AsyncClient, chat_client: AsyncMock):
    """Audio modality returns 503 when no speech client is configured."""
    client._transport.app.state.audio_speech = None  # type: ignore[union-attr]
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "modalities": ["text", "audio"],
            "audio": {"voice": "default"},
        },
    )
    assert resp.status_code == 503


async def test_chat_audio_synthesis_failure_continues(
    client: AsyncClient, chat_client: AsyncMock, speech_client: AsyncMock
):
    """Text streaming continues even when audio synthesis fails."""
    import json as _json

    speech_client.synthesize.side_effect = ConnectionError("TTS down")
    chat_client.chat_completions_stream_raw.return_value = _MockStreamResponse(_AUDIO_SSE_LINES)

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "modalities": ["text", "audio"],
            "audio": {"voice": "default", "format": "pcm16"},
        },
    )

    assert resp.status_code == 200

    text_tokens: list[str] = []
    audio_data_chunks: list[str] = []

    for raw_event in resp.text.strip().split("\n\n"):
        for line in raw_event.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            data = _json.loads(payload)
            delta = data["choices"][0]["delta"]
            if "content" in delta:
                text_tokens.append(delta["content"])
            if "audio" in delta and "data" in delta["audio"]:
                audio_data_chunks.append(delta["audio"]["data"])

    # Text still streamed even though TTS failed.
    assert text_tokens == ["Hello", "!", " How", " are", " you", "?"]
    # No audio data because synthesis raised.
    assert audio_data_chunks == []


# ===================================================================
# Text Completions
# ===================================================================


async def test_text_completions(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/completions returns a text completion response."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "my-model", "prompt": "Hello"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == "Hello world"
    chat_client.completions.assert_called_once_with(
        {"model": "my-model", "prompt": "Hello"},
    )


async def test_text_completions_stream(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/completions with stream=true returns an SSE stream."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "my-model", "prompt": "Hello", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    chat_client.completions_stream.assert_called_once()


async def test_text_completions_backend_unavailable(
    client: AsyncClient, chat_client: AsyncMock
):
    """Gateway returns 502 when the chat backend raises for completions."""
    chat_client.completions.side_effect = ConnectionError("Connection refused")
    resp = await client.post(
        "/v1/completions",
        json={"model": "my-model", "prompt": "Hello"},
    )
    assert resp.status_code == 502


async def test_text_completions_sampling_params(
    client: AsyncClient, chat_client: AsyncMock,
):
    """All sampling parameters for completions are forwarded."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "my-model",
            "prompt": "Once upon a time",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 50,
            "n": 3,
            "stop": [".", "!"],
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "best_of": 5,
            "echo": True,
            "suffix": " The end.",
            "seed": 123,
            "logprobs": 3,
            "user": "test-user",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.completions.call_args[0][0]
    assert call_args["temperature"] == 0.8
    assert call_args["top_p"] == 0.95
    assert call_args["max_tokens"] == 50
    assert call_args["n"] == 3
    assert call_args["stop"] == [".", "!"]
    assert call_args["presence_penalty"] == 0.2
    assert call_args["frequency_penalty"] == 0.1
    assert call_args["best_of"] == 5
    assert call_args["echo"] is True
    assert call_args["suffix"] == " The end."
    assert call_args["seed"] == 123
    assert call_args["logprobs"] == 3
    assert call_args["user"] == "test-user"


async def test_text_completions_array_prompt(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Completions accept array prompts."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "my-model",
            "prompt": ["Hello", "World"],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.completions.call_args[0][0]
    assert call_args["prompt"] == ["Hello", "World"]


async def test_text_completions_token_prompt(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Completions accept token ID array prompts."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "my-model",
            "prompt": [1, 2, 3, 4, 5],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.completions.call_args[0][0]
    assert call_args["prompt"] == [1, 2, 3, 4, 5]


async def test_text_completions_missing_model(client: AsyncClient):
    """Gateway returns 422 when 'model' is missing from completions."""
    resp = await client.post(
        "/v1/completions",
        json={"prompt": "Hello"},
    )
    assert resp.status_code == 422


async def test_text_completions_missing_prompt(client: AsyncClient):
    """Gateway returns 422 when 'prompt' is missing from completions."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "my-model"},
    )
    assert resp.status_code == 422


async def test_text_completions_extra_fields(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Unknown fields in completions pass through to the backend."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "my-model",
            "prompt": "Hello",
            "custom_setting": True,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.completions.call_args[0][0]
    assert call_args["custom_setting"] is True


# ===================================================================
# Responses API
# ===================================================================


async def test_responses(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/responses returns a response object."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["output"][0]["content"][0]["text"] == "Hello! How can I help?"
    chat_client.create_response.assert_called_once_with(
        {"model": "my-model", "input": "Hello"},
    )


async def test_responses_stream(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/responses with stream=true returns an SSE stream."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    chat_client.create_response_stream.assert_called_once_with(
        {"model": "my-model", "input": "Hello", "stream": True},
    )


async def test_responses_backend_unavailable(client: AsyncClient, chat_client: AsyncMock):
    """Gateway returns 502 when the chat backend raises for responses."""
    chat_client.create_response.side_effect = ConnectionError("Connection refused")
    resp = await client.post(
        "/v1/responses",
        json={"model": "my-model", "input": "Hello"},
    )
    assert resp.status_code == 502


async def test_responses_with_instructions(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Instructions are forwarded to the backend."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
            "instructions": "You are a pirate.",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args["instructions"] == "You are a pirate."


async def test_responses_with_tools(client: AsyncClient, chat_client: AsyncMock):
    """Responses API with tools returns function_call outputs."""
    chat_client.create_response.return_value = MOCK_RESPONSE_WITH_TOOL_CALLS

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What's the weather in SF?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["output"][0]["type"] == "function_call"
    assert data["output"][0]["name"] == "get_weather"

    # Verify tools were forwarded
    call_args = chat_client.create_response.call_args[0][0]
    assert len(call_args["tools"]) == 1
    assert call_args["tool_choice"] == "auto"


async def test_responses_with_input_items(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Input can be an array of message items."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": [
                {"type": "message", "role": "user", "content": "What's 2+2?"},
            ],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert isinstance(call_args["input"], list)
    assert call_args["input"][0]["role"] == "user"


async def test_responses_sampling_params(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Sampling parameters for responses are forwarded."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
            "temperature": 0.5,
            "top_p": 0.8,
            "max_output_tokens": 500,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args["temperature"] == 0.5
    assert call_args["top_p"] == 0.8
    assert call_args["max_output_tokens"] == 500


async def test_responses_missing_model(client: AsyncClient):
    """Gateway returns 422 when 'model' is missing from responses."""
    resp = await client.post(
        "/v1/responses",
        json={"input": "Hello"},
    )
    assert resp.status_code == 422


async def test_responses_missing_input(client: AsyncClient):
    """Gateway returns 422 when 'input' is missing from responses."""
    resp = await client.post(
        "/v1/responses",
        json={"model": "my-model"},
    )
    assert resp.status_code == 422


async def test_responses_extra_fields(client: AsyncClient, chat_client: AsyncMock):
    """Unknown fields in responses pass through to the backend."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
            "custom_param": "value",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args["custom_param"] == "value"


# ===================================================================
# Embeddings
# ===================================================================


async def test_embeddings(client: AsyncClient, chat_client: AsyncMock):
    """POST /v1/embeddings returns an embeddings response."""
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "my-model", "input": "Hello world"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    chat_client.embeddings.assert_called_once_with(
        {"model": "my-model", "input": "Hello world"},
    )


async def test_embeddings_backend_unavailable(client: AsyncClient, chat_client: AsyncMock):
    """Gateway returns 502 when the chat backend raises for embeddings."""
    chat_client.embeddings.side_effect = ConnectionError("Connection refused")
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "my-model", "input": "Hello world"},
    )
    assert resp.status_code == 502


async def test_embeddings_array_input(client: AsyncClient, chat_client: AsyncMock):
    """Embeddings accept array of strings."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "my-model",
            "input": ["Hello", "World"],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.embeddings.call_args[0][0]
    assert call_args["input"] == ["Hello", "World"]


async def test_embeddings_encoding_format(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Embeddings encoding_format is forwarded to the backend."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "my-model",
            "input": "Hello",
            "encoding_format": "base64",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.embeddings.call_args[0][0]
    assert call_args["encoding_format"] == "base64"


async def test_embeddings_dimensions(client: AsyncClient, chat_client: AsyncMock):
    """Embeddings dimensions parameter is forwarded."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "my-model",
            "input": "Hello",
            "dimensions": 256,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.embeddings.call_args[0][0]
    assert call_args["dimensions"] == 256


async def test_embeddings_missing_model(client: AsyncClient):
    """Gateway returns 422 when 'model' is missing from embeddings."""
    resp = await client.post(
        "/v1/embeddings",
        json={"input": "Hello"},
    )
    assert resp.status_code == 422


async def test_embeddings_missing_input(client: AsyncClient):
    """Gateway returns 422 when 'input' is missing from embeddings."""
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "my-model"},
    )
    assert resp.status_code == 422


async def test_embeddings_extra_fields(client: AsyncClient, chat_client: AsyncMock):
    """Unknown fields in embeddings pass through to the backend."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "model": "my-model",
            "input": "Hello",
            "custom_option": True,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.embeddings.call_args[0][0]
    assert call_args["custom_option"] is True


# ===================================================================
# No backend configured
# ===================================================================


async def test_no_backend_configured():
    """Gateway returns 503 for all endpoints when no backends are configured."""
    from fastapi import FastAPI

    from gateway_service.routes import completions, embeddings, responses
    from gateway_service.routes.chat import router as chat_router

    @asynccontextmanager
    async def noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(chat_router)
    test_app.include_router(completions.router)
    test_app.include_router(responses.router)
    test_app.include_router(embeddings.router)
    # No protocol slots set on state

    transport = ASGITransport(app=test_app)

    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503

    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/completions", json={"model": "x", "prompt": "hi"})
    assert resp.status_code == 503

    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/responses", json={"model": "x", "input": "hi"})
    assert resp.status_code == 503

    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/embeddings", json={"model": "x", "input": "hi"})
    assert resp.status_code == 503


# ===================================================================
# Unset fields are not forwarded
# ===================================================================


async def test_unset_fields_not_forwarded(client: AsyncClient, chat_client: AsyncMock):
    """Only fields the client explicitly sends appear in the backend payload."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    # Fields with defaults that weren't sent should be absent
    assert "temperature" not in call_args
    assert "top_p" not in call_args
    assert "tools" not in call_args
    assert "tool_choice" not in call_args
    assert "stream" not in call_args  # defaults to False, not sent
    assert "stream_options" not in call_args
    assert "response_format" not in call_args
    assert "seed" not in call_args
    # But the fields the user sent should be present
    assert "model" in call_args
    assert "messages" in call_args

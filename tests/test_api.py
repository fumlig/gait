from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.models import (
    ChatCompletionResponse,
    CompletionResponse,
    CreateResponseResponse,
    EmbeddingResponse,
    ModelObject,
    RawSegment,
    TranscriptionResult,
    WordTimestamp,
)
from gateway.providers.voice import VoiceClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway.models import SpeechRequest

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


MOCK_CHAT_COMPLETION = ChatCompletionResponse.model_validate({
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
})

MOCK_TOOL_CALL_RESPONSE = ChatCompletionResponse.model_validate({
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
})

MOCK_PARALLEL_TOOL_CALL_RESPONSE = ChatCompletionResponse.model_validate({
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
})

MOCK_COMPLETION = CompletionResponse.model_validate({
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
})

MOCK_RESPONSE = CreateResponseResponse.model_validate({
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
})

MOCK_RESPONSE_WITH_TOOL_CALLS = CreateResponseResponse.model_validate({
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
})

MOCK_EMBEDDINGS = EmbeddingResponse.model_validate({
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
})


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

    from gateway.routes import completions, embeddings, health, models, responses
    from gateway.routes.audio import router as audio_router
    from gateway.routes.chat import router as chat_router

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

    from gateway.routes import models

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

    from gateway.routes.audio import router as audio_router

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
    chat_client.chat_completions.assert_called_once()
    req = chat_client.chat_completions.call_args[0][0]
    assert req.model == "my-model"
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Hello"


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
    chat_client.chat_completions_stream.assert_called_once()
    req = chat_client.chat_completions_stream.call_args[0][0]
    assert req.model == "my-model"
    assert req.stream is True


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
    assert len(call_args.tools) == 1
    assert call_args.tools[0].function.name == "get_weather"
    assert call_args.tool_choice == "auto"


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
    messages = call_args.messages
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].content is None
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].id == "call_abc123"
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_abc123"


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
    assert call_args.parallel_tool_calls is True


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
    assert call_args.tool_choice["type"] == "function"
    assert call_args.tool_choice["function"]["name"] == "get_weather"


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
    assert call_args.tool_choice == "none"


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
    chat_client.chat_completions.return_value = ChatCompletionResponse.model_validate({
        "id": "chatcmpl-json",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "my-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer": 42}'},
                "finish_reason": "stop",
            }
        ],
    })

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
    assert call_args.response_format.type == "json_object"


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
    assert call_args.response_format.type == "json_schema"
    assert call_args.response_format.json_schema["name"] == "weather_response"


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
    assert call_args.temperature == 0.7
    assert call_args.top_p == 0.9
    assert call_args.max_tokens == 100
    assert call_args.max_completion_tokens == 200
    assert call_args.presence_penalty == 0.5
    assert call_args.frequency_penalty == 0.3
    assert call_args.seed == 42
    assert call_args.n == 2
    assert call_args.stop == ["\n", "END"]
    assert call_args.logprobs is True
    assert call_args.top_logprobs == 5
    assert call_args.user == "test-user"


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
    assert call_args.stream_options.include_usage is True


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
    assert call_args.logit_bias == {"50256": -100.0, "1": 5.0}


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
    assert call_args.messages[0].role == "system"
    assert call_args.messages[0].content == "You are a helpful assistant."


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
    content = call_args.messages[0].content
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
    assert call_args.messages[0].name == "Alice"


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
    assert call_args.custom_field == "custom_value"
    assert call_args.another_extension == 42


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
# Chat Completions — input_audio preprocessing
# ===================================================================


def _b64_wav() -> str:
    """Return base64-encoded minimal WAV for use in input_audio parts."""
    import base64 as _b64

    return _b64.b64encode(_WAV_HEADER).decode()


def _b64_pcm16() -> str:
    """Return base64-encoded raw PCM16 samples (two bytes of silence)."""
    import base64 as _b64

    return _b64.b64encode(b"\x00\x00").decode()


async def test_chat_input_audio_transcribed(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """input_audio content parts are transcribed and replaced with text."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64_wav(), "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    transcription_client.transcribe.assert_called_once()
    # The LLM should have received a text part instead of input_audio.
    req = chat_client.chat_completions.call_args[0][0]
    content = req.messages[0].content
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello world. This is a test."


async def test_chat_input_audio_pcm16_format(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """input_audio with format=pcm16 is converted to WAV before STT."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64_pcm16(), "format": "pcm16"},
                        }
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    transcription_client.transcribe.assert_called_once()
    # Verify we sent WAV (starts with RIFF) not raw PCM.
    call_kwargs = transcription_client.transcribe.call_args.kwargs
    assert call_kwargs["file"][:4] == b"RIFF"


async def test_chat_input_audio_mixed_content(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """Text parts are preserved alongside transcribed input_audio parts."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Listen to this:"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64_wav(), "format": "wav"},
                        },
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    req = chat_client.chat_completions.call_args[0][0]
    content = req.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Listen to this:"}
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "Hello world. This is a test."


async def test_chat_input_audio_no_stt_backend(
    client: AsyncClient,
    chat_client: AsyncMock,
):
    """input_audio returns 503 when no transcription backend is configured."""
    # Remove the transcription client
    client._transport.app.state.audio_transcriptions = None  # type: ignore[union-attr]
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64_wav(), "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 503


async def test_chat_input_audio_transcription_failure(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """input_audio returns 502 when the STT backend fails."""
    transcription_client.transcribe.side_effect = ConnectionError("STT down")
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64_wav(), "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 502


async def test_chat_input_audio_empty_data(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """input_audio with empty data returns 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "", "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


async def test_chat_input_audio_invalid_base64(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """input_audio with invalid base64 returns 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "!!!not-base64!!!", "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400
    assert "base64" in resp.json()["detail"].lower()


async def test_chat_input_audio_string_content_untouched(
    client: AsyncClient,
    chat_client: AsyncMock,
    transcription_client: AsyncMock,
):
    """Messages with plain string content are not affected by preprocessing."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    # Transcription should NOT have been called
    transcription_client.transcribe.assert_not_called()


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
    chat_client.completions.assert_called_once()
    req = chat_client.completions.call_args[0][0]
    assert req.model == "my-model"
    assert req.prompt == "Hello"


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
    assert call_args.temperature == 0.8
    assert call_args.top_p == 0.95
    assert call_args.max_tokens == 50
    assert call_args.n == 3
    assert call_args.stop == [".", "!"]
    assert call_args.presence_penalty == 0.2
    assert call_args.frequency_penalty == 0.1
    assert call_args.best_of == 5
    assert call_args.echo is True
    assert call_args.suffix == " The end."
    assert call_args.seed == 123
    assert call_args.logprobs == 3
    assert call_args.user == "test-user"


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
    assert call_args.prompt == ["Hello", "World"]


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
    assert call_args.prompt == [1, 2, 3, 4, 5]


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
    assert call_args.custom_setting is True


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
    chat_client.create_response.assert_called_once()
    req = chat_client.create_response.call_args[0][0]
    assert req.model == "my-model"
    assert req.input == "Hello"


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
    chat_client.create_response_stream.assert_called_once()
    req = chat_client.create_response_stream.call_args[0][0]
    assert req.model == "my-model"
    assert req.stream is True


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
    assert call_args.instructions == "You are a pirate."


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
    assert len(call_args.tools) == 1
    assert call_args.tool_choice == "auto"


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
    assert isinstance(call_args.input, list)
    assert call_args.input[0]["role"] == "user"


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
    assert call_args.temperature == 0.5
    assert call_args.top_p == 0.8
    assert call_args.max_output_tokens == 500


async def test_responses_sampling_params_extended(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Extended sampling parameters (top_k, min_p) are forwarded."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL",
            "input": "Hello",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args.model == "unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL"
    assert call_args.temperature == 0.6
    assert call_args.top_p == 0.95
    assert call_args.top_k == 20
    assert call_args.min_p == 0.0


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
    assert call_args.custom_param == "value"


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
    chat_client.embeddings.assert_called_once()
    req = chat_client.embeddings.call_args[0][0]
    assert req.model == "my-model"
    assert req.input == "Hello world"


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
    assert call_args.input == ["Hello", "World"]


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
    assert call_args.encoding_format == "base64"


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
    assert call_args.dimensions == 256


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
    assert call_args.custom_option is True


# ===================================================================
# No backend configured
# ===================================================================


async def test_no_backend_configured():
    """Gateway returns 503 for all endpoints when no backends are configured."""
    from fastapi import FastAPI

    from gateway.routes import completions, embeddings, responses
    from gateway.routes.chat import router as chat_router

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


async def test_unset_fields_excluded_from_dump(client: AsyncClient, chat_client: AsyncMock):
    """model_dump(exclude_unset=True) only includes fields the client sent."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert resp.status_code == 200
    req = chat_client.chat_completions.call_args[0][0]
    dump = req.model_dump(exclude_unset=True)
    # Fields with defaults that weren't sent should be absent from the dump
    assert "temperature" not in dump
    assert "top_p" not in dump
    assert "tools" not in dump
    assert "tool_choice" not in dump
    assert "stream" not in dump  # defaults to False, not sent
    assert "stream_options" not in dump
    assert "response_format" not in dump
    assert "seed" not in dump
    assert "reasoning_effort" not in dump
    # But the fields the user sent should be present
    assert "model" in dump
    assert "messages" in dump


# ===================================================================
# Chat Completions — reasoning
# ===================================================================

MOCK_REASONING_CHAT_COMPLETION = ChatCompletionResponse.model_validate({
    "id": "chatcmpl-reason-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think step by step about the meaning of life...",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 256,
        "total_tokens": 275,
        "completion_tokens_details": {
            "reasoning_tokens": 128,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {
            "cached_tokens": 0,
        },
    },
})


async def test_chat_completions_reasoning_effort(
    client: AsyncClient, chat_client: AsyncMock,
):
    """reasoning_effort is forwarded to the backend."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
            "reasoning_effort": "high",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args.reasoning_effort == "high"

    # Verify the dump only includes reasoning_effort when explicitly set
    dump = call_args.model_dump(exclude_unset=True)
    assert dump["reasoning_effort"] == "high"


async def test_chat_completions_reasoning_effort_levels(
    client: AsyncClient, chat_client: AsyncMock,
):
    """All reasoning effort levels (low, medium, high) are forwarded."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    for level in ("low", "medium", "high"):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "my-model",
                "messages": [{"role": "user", "content": "Think about this."}],
                "reasoning_effort": level,
            },
        )
        assert resp.status_code == 200
        call_args = chat_client.chat_completions.call_args[0][0]
        assert call_args.reasoning_effort == level


async def test_chat_completions_reasoning_usage(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Response includes completion_tokens_details with reasoning_tokens."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Think step by step."}],
            "reasoning_effort": "high",
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    # Verify usage structure
    usage = data["usage"]
    assert usage["prompt_tokens"] == 19
    assert usage["completion_tokens"] == 256
    assert usage["total_tokens"] == 275

    # Verify completion_tokens_details
    details = usage["completion_tokens_details"]
    assert details["reasoning_tokens"] == 128
    assert details["accepted_prediction_tokens"] == 0
    assert details["rejected_prediction_tokens"] == 0

    # Verify prompt_tokens_details
    prompt_details = usage["prompt_tokens_details"]
    assert prompt_details["cached_tokens"] == 0


async def test_chat_completions_reasoning_content_in_message(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Response message includes reasoning_content when the backend provides it."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Think about this."}],
            "reasoning_effort": "high",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is 42."
    assert msg["reasoning_content"] == "Let me think step by step about the meaning of life..."


async def test_chat_completions_reasoning_with_max_completion_tokens(
    client: AsyncClient, chat_client: AsyncMock,
):
    """max_completion_tokens is forwarded alongside reasoning_effort."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Think about this."}],
            "reasoning_effort": "medium",
            "max_completion_tokens": 16384,
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    assert call_args.reasoning_effort == "medium"
    assert call_args.max_completion_tokens == 16384


async def test_chat_completions_reasoning_not_sent_when_unset(
    client: AsyncClient, chat_client: AsyncMock,
):
    """reasoning_effort is excluded from the dump when not explicitly set."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    dump = call_args.model_dump(exclude_unset=True)
    assert "reasoning_effort" not in dump


async def test_chat_completions_reasoning_streaming(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Streaming with reasoning_effort returns SSE chunks with reasoning_content."""
    from starlette.responses import StreamingResponse

    def _sse(delta: dict, *, finish: str | None = None, usage: dict | None = None) -> bytes:
        obj = {
            "id": "chatcmpl-r1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "my-model",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish},
            ],
        }
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n".encode()

    async def _gen():
        yield _sse({"role": "assistant"})
        yield _sse({"reasoning_content": "Let me think"})
        yield _sse({"reasoning_content": " about this..."})
        yield _sse({"content": "The answer"})
        yield _sse({"content": " is 42."})
        yield _sse({}, finish="stop", usage={
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "total_tokens": 110,
            "completion_tokens_details": {"reasoning_tokens": 64},
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        yield b"data: [DONE]\n\n"

    chat_client.chat_completions_stream.return_value = StreamingResponse(
        _gen(), media_type="text/event-stream",
    )

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Think!"}],
            "stream": True,
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
        },
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Verify reasoning_effort was forwarded
    call_args = chat_client.chat_completions_stream.call_args[0][0]
    assert call_args.reasoning_effort == "high"

    # Parse SSE events
    events = []
    for raw in resp.text.strip().split("\n\n"):
        for line in raw.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))

    # Find reasoning_content deltas
    reasoning_parts = [
        e["choices"][0]["delta"]["reasoning_content"]
        for e in events
        if e["choices"][0]["delta"].get("reasoning_content")
    ]
    assert reasoning_parts == ["Let me think", " about this..."]

    # Find content deltas
    content_parts = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if e["choices"][0]["delta"].get("content")
    ]
    assert content_parts == ["The answer", " is 42."]

    # Find usage in the last event
    usage_event = [e for e in events if e.get("usage")]
    assert len(usage_event) == 1
    assert usage_event[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 64


async def test_chat_completions_reasoning_usage_without_details(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Normal completions without reasoning return usage without token details."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    usage = data["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 18
    # completion_tokens_details should not be present for non-reasoning models
    details = usage.get("completion_tokens_details")
    assert "completion_tokens_details" not in usage or details is None


async def test_chat_completions_reasoning_message_in_history(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Assistant messages with reasoning_content in history are forwarded."""
    chat_client.chat_completions.return_value = MOCK_REASONING_CHAT_COMPLETION

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "Simple addition: 2+2=4",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            "reasoning_effort": "low",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.chat_completions.call_args[0][0]
    messages = call_args.messages
    assert len(messages) == 3
    assert messages[1].reasoning_content == "Simple addition: 2+2=4"


# ===================================================================
# Responses API — reasoning
# ===================================================================

MOCK_REASONING_RESPONSE = CreateResponseResponse.model_validate({
    "id": "resp-reason-1",
    "object": "response",
    "created_at": 1700000000,
    "model": "my-model",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_001",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "I considered the mathematical properties of the number.",
                }
            ],
        },
        {
            "type": "message",
            "id": "msg-reason-1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The answer is 42."}],
            "status": "completed",
        },
    ],
    "status": "completed",
    "usage": {
        "input_tokens": 27,
        "output_tokens": 2076,
        "total_tokens": 2103,
        "output_tokens_details": {
            "reasoning_tokens": 1984,
        },
        "input_tokens_details": {
            "cached_tokens": 0,
        },
    },
})


async def test_responses_reasoning_effort(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Responses API with reasoning.effort is forwarded to the backend."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What is the meaning of life?",
            "reasoning": {"effort": "high"},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args.reasoning is not None
    assert call_args.reasoning.effort == "high"


async def test_responses_reasoning_effort_levels(
    client: AsyncClient, chat_client: AsyncMock,
):
    """All reasoning effort levels work in the Responses API."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    for level in ("low", "medium", "high"):
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "my-model",
                "input": "Think about this.",
                "reasoning": {"effort": level},
            },
        )
        assert resp.status_code == 200
        call_args = chat_client.create_response.call_args[0][0]
        assert call_args.reasoning.effort == level


async def test_responses_reasoning_summary(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Responses API with reasoning.summary is forwarded to the backend."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What is 2+2?",
            "reasoning": {"effort": "medium", "summary": "auto"},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args.reasoning.effort == "medium"
    assert call_args.reasoning.summary == "auto"

    # Verify the dump only includes set fields
    dump = call_args.model_dump(exclude_unset=True)
    assert dump["reasoning"]["effort"] == "medium"
    assert dump["reasoning"]["summary"] == "auto"


async def test_responses_reasoning_summary_levels(
    client: AsyncClient, chat_client: AsyncMock,
):
    """All reasoning summary levels work in the Responses API."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    for summary in ("auto", "concise", "detailed"):
        resp = await client.post(
            "/v1/responses",
            json={
                "model": "my-model",
                "input": "Think about this.",
                "reasoning": {"effort": "high", "summary": summary},
            },
        )
        assert resp.status_code == 200
        call_args = chat_client.create_response.call_args[0][0]
        assert call_args.reasoning.summary == summary


async def test_responses_reasoning_output_items(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Response includes reasoning output item with summary."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What is the meaning of life?",
            "reasoning": {"effort": "high", "summary": "auto"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    # First output item should be reasoning
    reasoning_item = data["output"][0]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["id"] == "rs_001"
    assert len(reasoning_item["summary"]) == 1
    assert reasoning_item["summary"][0]["type"] == "summary_text"
    assert "mathematical" in reasoning_item["summary"][0]["text"]

    # Second output item should be the message
    message_item = data["output"][1]
    assert message_item["type"] == "message"
    assert message_item["content"][0]["text"] == "The answer is 42."


async def test_responses_reasoning_usage(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Response usage includes output_tokens_details with reasoning_tokens."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Think about this.",
            "reasoning": {"effort": "high"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    usage = data["usage"]
    assert usage["input_tokens"] == 27
    assert usage["output_tokens"] == 2076
    assert usage["total_tokens"] == 2103

    # Verify output_tokens_details
    output_details = usage["output_tokens_details"]
    assert output_details["reasoning_tokens"] == 1984

    # Verify input_tokens_details
    input_details = usage["input_tokens_details"]
    assert input_details["cached_tokens"] == 0


async def test_responses_reasoning_not_sent_when_unset(
    client: AsyncClient, chat_client: AsyncMock,
):
    """reasoning is excluded from the dump when not explicitly set."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    dump = call_args.model_dump(exclude_unset=True)
    assert "reasoning" not in dump


async def test_responses_reasoning_usage_without_details(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Normal responses without reasoning return usage without token details."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    usage = data["usage"]
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 8
    assert usage["total_tokens"] == 18
    # Token details should not be present for non-reasoning responses
    assert "output_tokens_details" not in usage or usage.get("output_tokens_details") is None


async def test_responses_reasoning_streaming(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Streaming with reasoning returns SSE events including reasoning items."""
    from starlette.responses import StreamingResponse

    async def _gen():
        # Reasoning summary event
        yield (
            b'event: response.reasoning_summary_part.added\n'
            b'data: {"type":"response.reasoning_summary_part.added",'
            b'"item_id":"rs_001","part":{"type":"summary_text","text":""}}\n\n'
        )
        yield (
            b'event: response.reasoning_summary_text.delta\n'
            b'data: {"type":"response.reasoning_summary_text.delta",'
            b'"item_id":"rs_001","delta":"Thinking about math..."}\n\n'
        )
        yield (
            b'event: response.reasoning_summary_text.done\n'
            b'data: {"type":"response.reasoning_summary_text.done",'
            b'"item_id":"rs_001","text":"Thinking about math..."}\n\n'
        )
        # Message content event
        yield (
            b'event: response.output_text.delta\n'
            b'data: {"type":"response.output_text.delta",'
            b'"item_id":"msg_001","delta":"42"}\n\n'
        )
        yield (
            b'event: response.completed\n'
            b'data: {"type":"response.completed","response":{'
            b'"id":"resp-r1","object":"response","model":"my-model",'
            b'"output":[{"type":"reasoning","id":"rs_001","summary":'
            b'[{"type":"summary_text","text":"Thinking about math..."}]},'
            b'{"type":"message","id":"msg_001","role":"assistant",'
            b'"content":[{"type":"output_text","text":"42"}]}],'
            b'"status":"completed","usage":{"input_tokens":10,'
            b'"output_tokens":100,"total_tokens":110,'
            b'"output_tokens_details":{"reasoning_tokens":64},'
            b'"input_tokens_details":{"cached_tokens":0}}}}\n\n'
        )

    chat_client.create_response_stream.return_value = StreamingResponse(
        _gen(), media_type="text/event-stream",
    )

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What is the meaning of life?",
            "reasoning": {"effort": "high", "summary": "auto"},
            "stream": True,
        },
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Verify reasoning was forwarded
    call_args = chat_client.create_response_stream.call_args[0][0]
    assert call_args.reasoning.effort == "high"
    assert call_args.reasoning.summary == "auto"

    # Parse events — reasoning summary events should be present
    text = resp.text
    assert "reasoning_summary" in text
    assert "Thinking about math..." in text
    assert "reasoning_tokens" in text


async def test_responses_reasoning_with_tools(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Reasoning and tools can be combined in Responses API."""
    reasoning_with_tool = CreateResponseResponse.model_validate({
        "id": "resp-reason-tool-1",
        "object": "response",
        "created_at": 1700000000,
        "model": "my-model",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_002",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "I need to check the weather to answer this.",
                    }
                ],
            },
            {
                "type": "function_call",
                "id": "fc_002",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 40,
            "output_tokens": 300,
            "total_tokens": 340,
            "output_tokens_details": {"reasoning_tokens": 200},
            "input_tokens_details": {"cached_tokens": 0},
        },
    })
    chat_client.create_response.return_value = reasoning_with_tool

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What's the weather in SF?",
            "reasoning": {"effort": "medium"},
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    # First item is reasoning
    assert data["output"][0]["type"] == "reasoning"
    assert "weather" in data["output"][0]["summary"][0]["text"]

    # Second item is function_call
    assert data["output"][1]["type"] == "function_call"
    assert data["output"][1]["name"] == "get_weather"

    # Usage includes reasoning tokens
    assert data["usage"]["output_tokens_details"]["reasoning_tokens"] == 200


async def test_responses_reasoning_effort_only(
    client: AsyncClient, chat_client: AsyncMock,
):
    """reasoning with only effort (no summary) is forwarded correctly."""
    chat_client.create_response.return_value = MOCK_REASONING_RESPONSE

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "Hello",
            "reasoning": {"effort": "low"},
        },
    )

    assert resp.status_code == 200
    call_args = chat_client.create_response.call_args[0][0]
    assert call_args.reasoning.effort == "low"
    assert call_args.reasoning.summary is None

    # Only effort should appear in the dump
    dump = call_args.model_dump(exclude_unset=True)
    assert dump["reasoning"]["effort"] == "low"


async def test_responses_reasoning_multiple_summaries(
    client: AsyncClient, chat_client: AsyncMock,
):
    """Response can include multiple summary content blocks."""
    multi_summary = CreateResponseResponse.model_validate({
        "id": "resp-reason-multi",
        "object": "response",
        "created_at": 1700000000,
        "model": "my-model",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_multi",
                "summary": [
                    {"type": "summary_text", "text": "First, I considered the input."},
                    {"type": "summary_text", "text": "Then, I evaluated options."},
                ],
            },
            {
                "type": "message",
                "id": "msg-multi",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Here's my conclusion."}],
            },
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 15,
            "output_tokens": 500,
            "total_tokens": 515,
            "output_tokens_details": {"reasoning_tokens": 400},
        },
    })
    chat_client.create_response.return_value = multi_summary

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "my-model",
            "input": "What should I do?",
            "reasoning": {"effort": "high", "summary": "detailed"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    summaries = data["output"][0]["summary"]
    assert len(summaries) == 2
    assert summaries[0]["text"] == "First, I considered the input."
    assert summaries[1]["text"] == "Then, I evaluated options."

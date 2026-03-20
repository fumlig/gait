"""Resource protocols for OpenAI API endpoints.

Each protocol corresponds to one OpenAI resource group.  A backend client
implements the protocols for the resources it supports.  At startup the
gateway uses ``isinstance`` checks against these protocols to discover
what each client can handle and wires it to the matching ``app.state``
slot automatically.

The :data:`PROTOCOL_SLOTS` mapping connects each protocol to its
``app.state`` attribute name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import httpx
    from starlette.responses import StreamingResponse

    from gateway_service.models import SpeechRequest, TranscriptionResult, Voice


# ---------------------------------------------------------------------------
# LLM resource protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ChatCompletions(Protocol):
    """POST /v1/chat/completions — chat completion (streaming + non-streaming)."""

    async def chat_completions(self, body: dict) -> dict: ...
    async def chat_completions_stream(self, body: dict) -> StreamingResponse: ...
    async def chat_completions_stream_raw(self, body: dict) -> httpx.Response: ...


@runtime_checkable
class Completions(Protocol):
    """POST /v1/completions — legacy text completion."""

    async def completions(self, body: dict) -> dict: ...
    async def completions_stream(self, body: dict) -> StreamingResponse: ...


@runtime_checkable
class Responses(Protocol):
    """POST /v1/responses — OpenAI Responses API."""

    async def create_response(self, body: dict) -> dict: ...
    async def create_response_stream(self, body: dict) -> StreamingResponse: ...


@runtime_checkable
class Embeddings(Protocol):
    """POST /v1/embeddings — text embeddings."""

    async def embeddings(self, body: dict) -> dict: ...


# ---------------------------------------------------------------------------
# Audio resource protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioSpeech(Protocol):
    """POST /v1/audio/speech — text-to-speech synthesis."""

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]: ...


@runtime_checkable
class AudioTranscriptions(Protocol):
    """POST /v1/audio/transcriptions — speech-to-text transcription."""

    async def transcribe(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
        diarize: bool = False,
    ) -> TranscriptionResult: ...


@runtime_checkable
class AudioTranslations(Protocol):
    """POST /v1/audio/translations — speech-to-text translation."""

    async def translate(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> TranscriptionResult: ...


@runtime_checkable
class AudioVoices(Protocol):
    """Voice management CRUD — /v1/audio/voices."""

    async def list_voices(self) -> list[Voice]: ...
    async def get_voice(self, name: str) -> Voice: ...
    async def create_voice(self, name: str, audio_data: bytes) -> Voice: ...
    async def delete_voice(self, name: str) -> dict: ...


# ---------------------------------------------------------------------------
# Protocol → app.state slot mapping
# ---------------------------------------------------------------------------

PROTOCOL_SLOTS: list[tuple[type, str]] = [
    (ChatCompletions, "chat_completions"),
    (Completions, "completions"),
    (Responses, "responses"),
    (Embeddings, "embeddings"),
    (AudioSpeech, "audio_speech"),
    (AudioTranscriptions, "audio_transcriptions"),
    (AudioTranslations, "audio_translations"),
    (AudioVoices, "audio_voices"),
]

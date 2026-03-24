"""Resource protocols for OpenAI API endpoints.

Each protocol corresponds to one OpenAI resource group. Backend clients
implement the protocols they support. The gateway uses isinstance checks
to wire each client to the matching app.state slot automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.responses import StreamingResponse

    from gateway.models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        CompletionRequest,
        CompletionResponse,
        CreateResponseRequest,
        CreateResponseResponse,
        EmbeddingRequest,
        EmbeddingResponse,
        SpeechRequest,
        TranscriptionResult,
        Voice,
    )


@runtime_checkable
class ChatCompletions(Protocol):
    async def chat_completions(
        self, body: ChatCompletionRequest,
    ) -> ChatCompletionResponse: ...

    async def chat_completions_stream(
        self, body: ChatCompletionRequest,
    ) -> StreamingResponse: ...


@runtime_checkable
class Completions(Protocol):
    async def completions(
        self, body: CompletionRequest,
    ) -> CompletionResponse: ...

    async def completions_stream(
        self, body: CompletionRequest,
    ) -> StreamingResponse: ...


@runtime_checkable
class Responses(Protocol):
    async def create_response(
        self, body: CreateResponseRequest,
    ) -> CreateResponseResponse: ...

    async def create_response_stream(
        self, body: CreateResponseRequest,
    ) -> StreamingResponse: ...


@runtime_checkable
class Embeddings(Protocol):
    async def embeddings(
        self, body: EmbeddingRequest,
    ) -> EmbeddingResponse: ...


@runtime_checkable
class AudioSpeech(Protocol):
    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]: ...
    async def synthesize_stream(self, request: SpeechRequest) -> StreamingResponse: ...


@runtime_checkable
class AudioTranscriptions(Protocol):
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

    def transcribe_stream(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> AsyncIterator[dict]: ...


@runtime_checkable
class AudioTranslations(Protocol):
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
    async def list_voices(self) -> list[Voice]: ...
    async def get_voice(self, name: str) -> Voice: ...
    async def create_voice(self, name: str, audio_data: bytes) -> Voice: ...
    async def delete_voice(self, name: str) -> dict[str, Any]: ...


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

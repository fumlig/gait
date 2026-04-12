"""POST /v1/audio/speech — proxied to the speech provider.

When ``stream_format`` is ``"sse"``, audio is returned as typed
``SpeechStreamEvent`` models serialized into SSE frames using FastAPI's
``EventSourceResponse`` and ``format_sse_event``, matching the pattern
used by the responses and transcriptions routes.

Format conversion is handled entirely by the gateway.  Each provider
declares ``native_audio_format`` (e.g. WAV for chatterbox); the gateway
calls ``convert_audio`` to translate to whatever ``response_format`` the
client requested.

For formats that are streamable without buffering (WAV and PCM when the
backend produces WAV natively), the response is streamed chunk-by-chunk.
All other formats require full buffering for container encoding.
"""

from __future__ import annotations

import base64
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import EventSourceResponse, Response
from fastapi.sse import format_sse_event
from pydantic import TypeAdapter
from starlette.responses import StreamingResponse

from gateway.deps import SpeechClient, backend_errors
from gateway.formatting import convert_audio, wav_to_pcm_stream_chunk
from gateway.models import (
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechAudioUsage,
    SpeechRequest,
    SpeechResponseFormat,
    SpeechStreamFormat,
)
from gateway.models.audio import SpeechStreamEvent
from gateway.text_preprocessing import preprocess_speech_text

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway.providers.protocols import AudioSpeech

_stream_event_adapter = TypeAdapter(SpeechStreamEvent)

router = APIRouter()


_FORMAT_CONTENT_TYPES: dict[SpeechResponseFormat, str] = {
    SpeechResponseFormat.mp3: "audio/mpeg",
    SpeechResponseFormat.wav: "audio/wav",
    SpeechResponseFormat.opus: "audio/opus",
    SpeechResponseFormat.aac: "audio/aac",
    SpeechResponseFormat.flac: "audio/flac",
    SpeechResponseFormat.pcm: "audio/pcm",
}

# Formats that can be streamed chunk-by-chunk from a WAV backend without
# full buffering: WAV itself (passthrough) and PCM (strip headers).
_STREAMABLE_FROM_WAV: frozenset[SpeechResponseFormat] = frozenset({
    SpeechResponseFormat.wav,
    SpeechResponseFormat.pcm,
})

# SSE chunk size for base64-encoded audio (16 KiB of raw audio per event).
_SSE_CHUNK_SIZE = 16 * 1024


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def _serialize_events(
    events: AsyncIterator[SpeechStreamEvent],
) -> AsyncIterator[bytes]:
    """Serialize typed speech events into SSE wire format."""
    async for event in events:
        data_str = _stream_event_adapter.dump_json(event).decode()
        yield format_sse_event(data_str=data_str, event=event.type)
    yield format_sse_event(data_str="[DONE]")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_instructions(body: SpeechRequest, client: AudioSpeech) -> None:
    """Raise 400 if the client sent ``instructions`` to a provider that
    does not support them."""
    if body.instructions is not None and not client.supports_instructions:
        raise HTTPException(
            status_code=400,
            detail=(
                "The 'instructions' field is not supported by the current"
                " speech provider. Remove it from the request."
            ),
        )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/v1/audio/speech", response_model=None)
async def create_speech(
    body: SpeechRequest, client: SpeechClient,
) -> Response | EventSourceResponse | StreamingResponse:
    body.input = preprocess_speech_text(body.input)
    _validate_instructions(body, client)

    fmt = body.response_format
    native = client.native_audio_format
    sse = body.stream_format == SpeechStreamFormat.sse

    if sse:
        return await _create_speech_sse(body, client, fmt, native)

    # Streamable: formats where we can proxy chunks without full buffering.
    if native == SpeechResponseFormat.wav and fmt in _STREAMABLE_FROM_WAV:
        return await _create_speech_streaming(body, client, fmt)

    # Everything else: buffer, convert, return.
    return await _create_speech_buffered(body, client, fmt, native)


# ---------------------------------------------------------------------------
# Non-SSE response paths
# ---------------------------------------------------------------------------


async def _create_speech_streaming(
    body: SpeechRequest,
    client: AudioSpeech,
    fmt: SpeechResponseFormat,
) -> StreamingResponse:
    """Stream audio chunk-by-chunk for WAV and PCM."""
    async with backend_errors("Speech synthesis"):
        upstream = await client.synthesize_stream(body)

    if fmt == SpeechResponseFormat.wav:
        return upstream  # passthrough

    # PCM: strip WAV headers from the streamed chunks.
    content_type = _FORMAT_CONTENT_TYPES[fmt]

    async def _strip_headers() -> AsyncIterator[bytes]:
        async for chunk in upstream.body_iterator:
            if isinstance(chunk, str):
                raw = chunk.encode("latin-1")
            else:
                raw = bytes(chunk)
            pcm = wav_to_pcm_stream_chunk(raw)
            if pcm:
                yield pcm

    return StreamingResponse(
        _strip_headers(),
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="speech.{fmt.value}"'},
    )


async def _create_speech_buffered(
    body: SpeechRequest,
    client: AudioSpeech,
    fmt: SpeechResponseFormat,
    native: SpeechResponseFormat,
) -> Response:
    """Buffer the full audio, convert format, return as a single response."""
    async with backend_errors("Speech synthesis"):
        raw_bytes, _ = await client.synthesize(body)

    audio_bytes = convert_audio(
        raw_bytes,
        source_format=native.value,
        target_format=fmt.value,
    )
    content_type = _FORMAT_CONTENT_TYPES.get(fmt, "application/octet-stream")

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": f'inline; filename="speech.{fmt.value}"',
        },
    )


# ---------------------------------------------------------------------------
# SSE response path
# ---------------------------------------------------------------------------


async def _create_speech_sse(
    body: SpeechRequest,
    client: AudioSpeech,
    fmt: SpeechResponseFormat,
    native: SpeechResponseFormat,
) -> EventSourceResponse:
    """Return audio wrapped in Server-Sent Events."""
    async with backend_errors("Speech synthesis"):
        raw_bytes, _ = await client.synthesize(body)

    audio_bytes = convert_audio(
        raw_bytes,
        source_format=native.value,
        target_format=fmt.value,
    )

    # Rough token estimate: count whitespace-separated words in the input.
    input_tokens = len(body.input.split())
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    async def _generate_events() -> AsyncIterator[SpeechStreamEvent]:
        offset = 0
        while offset < len(audio_bytes):
            chunk = audio_bytes[offset : offset + _SSE_CHUNK_SIZE]
            offset += _SSE_CHUNK_SIZE
            b64 = base64.b64encode(chunk).decode("ascii")
            yield SpeechAudioDeltaEvent(response_id=response_id, audio=b64)

        yield SpeechAudioDoneEvent(
            response_id=response_id,
            usage=SpeechAudioUsage(
                input_tokens=input_tokens,
                output_tokens=0,
                total_tokens=input_tokens,
            ),
        )

    return EventSourceResponse(_serialize_events(_generate_events()))

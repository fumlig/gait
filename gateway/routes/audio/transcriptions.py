"""POST /v1/audio/transcriptions — proxied to the transcription provider.

Streaming transcriptions are received as typed
``TranscriptionStreamEvent`` models from the provider and serialized
into SSE frames using FastAPI's ``EventSourceResponse`` and
``format_sse_event``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import EventSourceResponse
from fastapi.sse import format_sse_event
from pydantic import TypeAdapter

from gateway.deps import TranscriptionClient, backend_errors
from gateway.formatting import format_transcription
from gateway.models import TranscriptionResponseFormat
from gateway.models.audio import TranscriptionStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.responses import Response

_stream_event_adapter = TypeAdapter(TranscriptionStreamEvent)

router = APIRouter()


async def _serialize_events(
    events: AsyncIterator[TranscriptionStreamEvent],
) -> AsyncIterator[bytes]:
    """Serialize typed transcription events into SSE wire format."""
    async for event in events:
        data_str = _stream_event_adapter.dump_json(event).decode()
        yield format_sse_event(data_str=data_str, event=event.type)
    yield format_sse_event(data_str="[DONE]")


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    client: TranscriptionClient,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarize: str = Form("false"),
    stream: str = Form("false"),
) -> Response | EventSourceResponse:
    want_stream = stream.lower() == "true"

    if not want_stream:
        try:
            fmt = TranscriptionResponseFormat(response_format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid response_format '{response_format}'."
                    " Use one of: json, text, srt, verbose_json, vtt."
                ),
            ) from None

    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    filename = file.filename or "audio.wav"
    want_diarize = diarize.lower() == "true"

    want_words = (
        not want_stream
        and (
            fmt == TranscriptionResponseFormat.verbose_json
            or (timestamp_granularities is not None and "word" in timestamp_granularities)
            or want_diarize  # diarization needs word alignment
        )
    )

    if want_stream:
        return EventSourceResponse(
            _serialize_events(
                client.transcribe_stream(
                    file=audio_data,
                    filename=filename,
                    model=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                ),
            ),
        )

    async with backend_errors("Transcription"):
        result = await client.transcribe(
            file=audio_data,
            filename=filename,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=want_words,
            diarize=want_diarize,
        )

    return format_transcription(result, fmt, task="transcribe")

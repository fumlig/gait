from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from gateway.deps import TranscriptionClient, backend_errors
from gateway.formatting import format_transcription
from gateway.models import TranscriptionResponseFormat

if TYPE_CHECKING:
    from starlette.responses import Response

router = APIRouter()


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
) -> Response:
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
        fmt == TranscriptionResponseFormat.verbose_json
        or (timestamp_granularities is not None and "word" in timestamp_granularities)
        or want_diarize  # diarization needs word alignment
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

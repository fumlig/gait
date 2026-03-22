from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from gateway.deps import TranslationClient, backend_errors
from gateway.formatting import format_transcription
from gateway.models import TranscriptionResponseFormat

if TYPE_CHECKING:
    from starlette.responses import Response

router = APIRouter()


@router.post("/v1/audio/translations", response_model=None)
async def create_translation(
    client: TranslationClient,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
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
    want_words = fmt == TranscriptionResponseFormat.verbose_json

    async with backend_errors("Translation"):
        result = await client.translate(
            file=audio_data,
            filename=filename,
            model=model,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=want_words,
        )

    return format_transcription(result, fmt, task="translate")

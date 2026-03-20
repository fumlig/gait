from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from gateway_service.formatting import format_transcription
from gateway_service.models import TranscriptionResponseFormat

if TYPE_CHECKING:
    from starlette.responses import Response

    from gateway_service.providers.protocols import AudioTranslations

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_translation_client(request: Request) -> AudioTranslations:
    client = getattr(request.app.state, "audio_translations", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No translation backend configured.")
    return client


@router.post("/v1/audio/translations", response_model=None)
async def create_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
) -> Response:
    client = _get_translation_client(request)

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

    try:
        result = await client.translate(
            file=audio_data,
            filename=filename,
            model=model,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=want_words,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(status_code=502, detail="Translation backend unavailable.") from exc

    return format_transcription(result, fmt, task="translate")

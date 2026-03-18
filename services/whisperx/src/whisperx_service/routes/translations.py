"""POST /v1/audio/translations -- OpenAI-compatible translation endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from whisperx_service.config import settings
from whisperx_service.engine import engine
from whisperx_service.models import ResponseFormat
from whisperx_service.routes.transcriptions import _format_response

if TYPE_CHECKING:
    from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
) -> Response:
    """Translate audio into English text (OpenAI-compatible).

    This endpoint forces task=translate, producing English output regardless
    of the input language.
    """
    # Validate response format -- translations support json, text, srt, verbose_json, vtt
    try:
        fmt = ResponseFormat(response_format)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid response_format '{response_format}'. "
                f"Use one of: json, text, srt, verbose_json, vtt."
            ),
        ) from None

    # Ensure model is loaded (dynamic swap)
    try:
        engine.ensure_model(model)
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Read and validate file
    audio_data = await file.read()
    if len(audio_data) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size} bytes.",
        )
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    want_words = fmt == ResponseFormat.verbose_json

    # Run translation (task=translate forces English output)
    try:
        result = engine.transcribe(
            audio_data,
            prompt=prompt,
            temperature=temperature,
            task="translate",
            word_timestamps=want_words,
            diarize=False,
        )
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail="Translation failed.") from exc

    return _format_response(result, fmt, task="translate")

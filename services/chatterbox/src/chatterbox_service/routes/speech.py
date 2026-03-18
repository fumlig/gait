"""POST /v1/audio/speech — OpenAI-compatible TTS endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from chatterbox_service.audio import encode_audio, stream_bytes
from chatterbox_service.engine import engine
from chatterbox_service.models import CONTENT_TYPE_MAP, SpeechRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> StreamingResponse:
    """Generate speech audio from text input."""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Validate model name
    if req.model not in ("chatterbox-turbo", "tts-1", "tts-1-hd"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: chatterbox-turbo",
        )

    # Validate voice
    available_voices = engine.list_voices()
    if available_voices and req.voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Available: {', '.join(available_voices)}",
        )

    # Generate
    try:
        wav, sr = engine.generate(text=req.input, voice=req.voice, speed=req.speed)
    except Exception as exc:
        logger.exception("Speech generation failed")
        raise HTTPException(status_code=500, detail="Speech generation failed.") from exc

    # Encode to requested format
    audio_bytes = encode_audio(wav, sr, req.response_format)
    content_type = CONTENT_TYPE_MAP[req.response_format]

    return StreamingResponse(
        stream_bytes(audio_bytes),
        media_type=content_type,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": f'inline; filename="speech.{req.response_format.value}"',
        },
    )

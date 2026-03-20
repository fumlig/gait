"""POST /v1/audio/speech — OpenAI-compatible TTS endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from gateway_service.models import SpeechRequest, SpeechResponseFormat

if TYPE_CHECKING:
    from gateway_service.clients.protocols import AudioSpeech

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_speech_client(request: Request) -> AudioSpeech:
    """Resolve the speech client from app state."""
    client = getattr(request.app.state, "audio_speech", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No speech backend configured.")
    return client


# Map gateway SpeechResponseFormat enum to MIME content types.
_FORMAT_CONTENT_TYPES: dict[SpeechResponseFormat, str] = {
    SpeechResponseFormat.mp3: "audio/mpeg",
    SpeechResponseFormat.wav: "audio/wav",
    SpeechResponseFormat.opus: "audio/opus",
    SpeechResponseFormat.aac: "audio/aac",
    SpeechResponseFormat.flac: "audio/flac",
    SpeechResponseFormat.pcm: "audio/pcm",
}


@router.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request) -> Response:
    """Generate speech audio from text input."""
    client = _get_speech_client(request)

    try:
        wav_bytes, _ = await client.synthesize(body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Speech synthesis failed")
        raise HTTPException(status_code=502, detail="Speech backend unavailable.") from exc

    # Convert WAV to the requested output format
    audio_bytes, content_type = _convert_audio(wav_bytes, body.response_format)

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": f'inline; filename="speech.{body.response_format.value}"',
        },
    )


def _convert_audio(wav_bytes: bytes, fmt: SpeechResponseFormat) -> tuple[bytes, str]:
    """Convert raw WAV bytes to the requested output format.

    Currently supports WAV (passthrough) and MP3 (via pydub/ffmpeg).
    Other formats raise 400.
    """
    content_type = _FORMAT_CONTENT_TYPES.get(fmt, "application/octet-stream")

    if fmt == SpeechResponseFormat.wav:
        return wav_bytes, content_type

    if fmt == SpeechResponseFormat.mp3:
        from gateway_service.formatting import wav_to_mp3

        return wav_to_mp3(wav_bytes), content_type

    # Formats not yet supported by any backend
    raise HTTPException(
        status_code=400,
        detail=f"Audio format '{fmt.value}' is not currently supported. Use 'mp3' or 'wav'.",
    )

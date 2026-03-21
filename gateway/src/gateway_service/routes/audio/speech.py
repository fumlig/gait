from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from gateway_service.deps import SpeechClient, backend_errors
from gateway_service.models import SpeechRequest, SpeechResponseFormat
from gateway_service.text_preprocessing import preprocess_speech_text

router = APIRouter()


_FORMAT_CONTENT_TYPES: dict[SpeechResponseFormat, str] = {
    SpeechResponseFormat.mp3: "audio/mpeg",
    SpeechResponseFormat.wav: "audio/wav",
    SpeechResponseFormat.opus: "audio/opus",
    SpeechResponseFormat.aac: "audio/aac",
    SpeechResponseFormat.flac: "audio/flac",
    SpeechResponseFormat.pcm: "audio/pcm",
}


@router.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, client: SpeechClient) -> Response:
    body.input = preprocess_speech_text(body.input)

    async with backend_errors("Speech synthesis"):
        wav_bytes, _ = await client.synthesize(body)

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
    """Convert WAV to the requested format. Supports WAV (passthrough) and MP3."""
    content_type = _FORMAT_CONTENT_TYPES.get(fmt, "application/octet-stream")

    if fmt == SpeechResponseFormat.wav:
        return wav_bytes, content_type

    if fmt == SpeechResponseFormat.mp3:
        from gateway_service.formatting import wav_to_mp3
        return wav_to_mp3(wav_bytes), content_type

    raise HTTPException(
        status_code=400,
        detail=f"Audio format '{fmt.value}' is not currently supported. Use 'mp3' or 'wav'.",
    )

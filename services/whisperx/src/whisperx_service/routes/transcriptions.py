"""POST /v1/audio/transcriptions -- OpenAI-compatible STT endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response

from whisperx_service.config import settings
from whisperx_service.engine import engine
from whisperx_service.models import (
    ResponseFormat,
    Segment,
    TimestampGranularity,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    WordTimestamp,
)

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

router = APIRouter()

ACCEPTED_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/mpga",
    "audio/x-m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "video/mp4",
    "video/mpeg",
    "video/webm",
    "application/octet-stream",  # allow generic binary
}


def _parse_granularities(granularities: list[str] | None) -> set[TimestampGranularity]:
    """Parse and validate timestamp_granularities."""
    if not granularities:
        return set()
    result = set()
    for g in granularities:
        try:
            result.add(TimestampGranularity(g))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timestamp granularity '{g}'. Use 'word' or 'segment'.",
            ) from None
    return result


def _build_segment(raw: dict[str, Any], idx: int) -> Segment:
    """Convert a raw whisperx segment dict to a Segment schema."""
    words = []
    for w in raw.get("words", []):
        if "start" in w and "end" in w:
            words.append(
                WordTimestamp(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    score=w.get("score"),
                )
            )
    return Segment(
        id=idx,
        start=raw.get("start", 0.0),
        end=raw.get("end", 0.0),
        text=raw.get("text", "").strip(),
        words=words,
        speaker=raw.get("speaker"),
    )


def _segments_to_srt(segments: list[Segment]) -> str:
    """Format segments as SRT subtitles."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list[Segment]) -> str:
    """Format segments as WebVTT subtitles."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for VTT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(
        None, alias="timestamp_granularities[]"
    ),
) -> Response:
    """Transcribe audio into text (OpenAI-compatible)."""
    # Validate response format
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

    # Parse granularities
    granularities = _parse_granularities(timestamp_granularities)

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

    # Determine if we need word timestamps or diarization
    want_words = (
        fmt == ResponseFormat.verbose_json or TimestampGranularity.word in granularities
    )
    want_diarize = fmt == ResponseFormat.verbose_json and settings.enable_diarization

    # Run transcription
    try:
        result = engine.transcribe(
            audio_data,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task="transcribe",
            word_timestamps=want_words,
            diarize=want_diarize,
        )
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed.") from exc

    return _format_response(result, fmt, task="transcribe")


def _format_response(
    result: dict[str, Any], fmt: ResponseFormat, task: str
) -> Response:
    """Format the transcription result according to the requested format."""
    segments = [_build_segment(seg, i) for i, seg in enumerate(result["segments"])]
    text = result["text"]

    if fmt == ResponseFormat.text:
        return PlainTextResponse(content=text)

    if fmt == ResponseFormat.json:
        return Response(
            content=TranscriptionResponse(text=text).model_dump_json(),
            media_type="application/json",
        )

    if fmt == ResponseFormat.verbose_json:
        all_words = []
        for seg in segments:
            all_words.extend(seg.words)
        resp = VerboseTranscriptionResponse(
            task=task,
            language=result.get("language", ""),
            duration=result.get("duration", 0.0),
            text=text,
            words=all_words,
            segments=segments,
        )
        return Response(
            content=resp.model_dump_json(),
            media_type="application/json",
        )

    if fmt == ResponseFormat.srt:
        return PlainTextResponse(
            content=_segments_to_srt(segments),
            media_type="text/plain",
        )

    if fmt == ResponseFormat.vtt:
        return PlainTextResponse(
            content=_segments_to_vtt(segments),
            media_type="text/plain",
        )

    # Should not happen due to validation above
    return Response(
        content=TranscriptionResponse(text=text).model_dump_json(),
        media_type="application/json",
    )

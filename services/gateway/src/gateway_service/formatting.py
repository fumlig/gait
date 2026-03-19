"""Response formatting utilities for gateway routes.

Contains STT response formatting (json, text, srt, vtt, verbose_json) and
TTS audio format conversion (WAV to MP3).
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from fastapi.responses import PlainTextResponse, Response

from gateway_service.models import (
    Segment,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
    WordTimestamp,
)

if TYPE_CHECKING:
    from gateway_service.models import TranscriptionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STT response formatting
# ---------------------------------------------------------------------------


def format_transcription(
    result: TranscriptionResult,
    fmt: TranscriptionResponseFormat,
    task: str = "transcribe",
) -> Response:
    """Format a ``TranscriptionResult`` into the OpenAI-compatible response shape.

    The *fmt* parameter determines the output format:
    - json: ``{"text": "..."}``
    - text: plain text
    - verbose_json: full segments, words, language, duration
    - srt: SRT subtitles
    - vtt: WebVTT subtitles
    """
    # Convert RawSegments to OpenAI-compatible Segments
    segments = [
        Segment(
            id=i,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=[
                WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    score=w.score,
                )
                for w in seg.words
            ],
            speaker=seg.speaker,
        )
        for i, seg in enumerate(result.segments)
    ]

    if fmt == TranscriptionResponseFormat.text:
        return PlainTextResponse(content=result.text)

    if fmt == TranscriptionResponseFormat.json:
        return Response(
            content=TranscriptionResponse(text=result.text).model_dump_json(),
            media_type="application/json",
        )

    if fmt == TranscriptionResponseFormat.verbose_json:
        all_words = []
        for seg in segments:
            all_words.extend(seg.words)
        resp = VerboseTranscriptionResponse(
            task=task,
            language=result.language,
            duration=result.duration,
            text=result.text,
            words=all_words,
            segments=segments,
        )
        return Response(
            content=resp.model_dump_json(),
            media_type="application/json",
        )

    if fmt == TranscriptionResponseFormat.srt:
        return PlainTextResponse(
            content=_segments_to_srt(segments),
            media_type="text/plain",
        )

    if fmt == TranscriptionResponseFormat.vtt:
        return PlainTextResponse(
            content=_segments_to_vtt(segments),
            media_type="text/plain",
        )

    # Fallback (should not happen due to enum validation)
    return Response(
        content=TranscriptionResponse(text=result.text).model_dump_json(),
        media_type="application/json",
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


# ---------------------------------------------------------------------------
# TTS audio format conversion
# ---------------------------------------------------------------------------


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 bytes using pydub (requires ffmpeg)."""
    from pydub import AudioSegment

    wav_buf = io.BytesIO(wav_bytes)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3")
    return mp3_buf.getvalue()

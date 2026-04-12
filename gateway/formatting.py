"""STT response formatting (json, text, srt, vtt, verbose_json) and
TTS audio format conversion.

All audio conversion goes through pydub (which shells out to ffmpeg).
The ``convert_audio`` function is the single entry-point used by the
gateway speech route for translating between whatever the backend
produces natively and the format the client requested.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

from fastapi.responses import PlainTextResponse, Response

from gateway.models import (
    Segment,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
    WordTimestamp,
)

if TYPE_CHECKING:
    from gateway.models import TranscriptionResult

logger = logging.getLogger(__name__)


def format_transcription(
    result: TranscriptionResult,
    fmt: TranscriptionResponseFormat,
    task: str = "transcribe",
) -> Response:
    """Format a TranscriptionResult into the requested response shape."""
    segments = [
        Segment(
            id=i,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=[
                WordTimestamp(word=w.word, start=w.start, end=w.end, score=w.score)
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

    return Response(
        content=TranscriptionResponse(text=result.text).model_dump_json(),
        media_type="application/json",
    )


def _segments_to_srt(segments: list[Segment]) -> str:
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
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# TTS audio conversion
# ---------------------------------------------------------------------------

# OpenAI spec: PCM is "24kHz (16-bit signed, low-endian), without the header".
_PCM_SAMPLE_RATE = 24000
_PCM_SAMPLE_WIDTH = 2  # 16-bit
_PCM_CHANNELS = 1


def _load_audio(audio_bytes: bytes, source_format: str) -> Any:
    """Load raw audio bytes into a pydub ``AudioSegment``."""
    from pydub import AudioSegment

    buf = io.BytesIO(audio_bytes)
    return AudioSegment.from_file(buf, format=source_format)


def _export_audio(segment: Any, fmt: str, **kwargs: Any) -> bytes:
    """Export a pydub ``AudioSegment`` to the given container format."""
    buf = io.BytesIO()
    segment.export(buf, format=fmt, **kwargs)
    return buf.getvalue()


def convert_audio(
    audio_bytes: bytes,
    *,
    source_format: str,
    target_format: str,
) -> bytes:
    """Convert *audio_bytes* from *source_format* to *target_format*.

    Both format strings use the ``SpeechResponseFormat`` enum values
    (``"wav"``, ``"mp3"``, ``"opus"``, ``"flac"``, ``"aac"``,
    ``"pcm"``).

    PCM output follows the OpenAI spec: 24 kHz, 16-bit signed LE,
    mono, headerless.
    """
    if source_format == target_format:
        return audio_bytes

    segment = _load_audio(audio_bytes, source_format)

    if target_format == "pcm":
        # Conform to OpenAI PCM spec: 24 kHz, 16-bit, mono.
        segment = (
            segment
            .set_frame_rate(_PCM_SAMPLE_RATE)
            .set_channels(_PCM_CHANNELS)
            .set_sample_width(_PCM_SAMPLE_WIDTH)
        )
        return segment.raw_data

    if target_format == "aac":
        # ffmpeg writes AAC into an ADTS container.
        return _export_audio(segment, "adts")

    if target_format == "opus":
        # ffmpeg encodes Opus inside an OGG container.
        return _export_audio(segment, "opus", codec="libopus")

    # mp3, wav, flac — pydub/ffmpeg handles directly.
    return _export_audio(segment, target_format)


def wav_to_pcm_stream_chunk(wav_chunk: bytes) -> bytes:
    """Strip WAV headers from a chunk for raw PCM streaming.

    For the first chunk (containing a RIFF header), skips to the
    ``data`` sub-chunk payload.  Subsequent chunks (pure sample data)
    are returned as-is.
    """
    if wav_chunk[:4] != b"RIFF":
        return wav_chunk

    # Walk the WAV to find the 'data' sub-chunk.
    pos = 12  # skip RIFF header (4 + 4 + 4)
    while pos + 8 <= len(wav_chunk):
        chunk_id = wav_chunk[pos : pos + 4]
        chunk_size = int.from_bytes(wav_chunk[pos + 4 : pos + 8], "little")
        if chunk_id == b"data":
            return wav_chunk[pos + 8 :]
        pos += 8 + chunk_size
    # Fallback: return everything after a 44-byte standard header.
    return wav_chunk[44:] if len(wav_chunk) > 44 else wav_chunk

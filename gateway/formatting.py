"""STT response formatting (json, text, srt, vtt, verbose_json) and
TTS/STT audio conversion (WAV ↔ PCM16, WAV → MP3).
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING

from fastapi.responses import PlainTextResponse, Response
from starlette.responses import StreamingResponse

from gateway.models import (
    Segment,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
    WordTimestamp,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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


def stream_transcription(result: TranscriptionResult) -> StreamingResponse:
    """Return a TranscriptionResult as an OpenAI-compatible SSE stream.

    Emits one ``transcript.text.delta`` per segment, then a final
    ``transcript.text.done`` with the complete text.  This lets a
    batch STT backend (whisperx) satisfy clients that request
    ``stream=true``.
    """

    async def _generate() -> AsyncIterator[str]:
        for seg in result.segments:
            delta_event = {
                "type": "transcript.text.delta",
                "delta": seg.text,
            }
            yield f"event: transcript.text.delta\ndata: {json.dumps(delta_event)}\n\n"

        done_event = {
            "type": "transcript.text.done",
            "text": result.text,
        }
        yield f"event: transcript.text.done\ndata: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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


def pcm16_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM16 samples in a WAV container.

    Inverse of :func:`wav_to_pcm16`.  Used to convert ``input_audio``
    content parts (format ``pcm16``) into WAV before sending to the
    STT backend.
    """
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes, target_sr: int = 24000) -> tuple[bytes, int]:
    """Convert WAV bytes to raw PCM16 mono at a target sample rate.

    Returns (pcm_bytes, sample_rate). Falls back to pydub for non-standard
    WAV formats (e.g. 32-bit float).
    """
    import wave

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width == 2 and channels == 1 and sr == target_sr:
            return frames, sr
    except wave.Error:
        pass

    from pydub import AudioSegment

    segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    segment = segment.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
    return segment.raw_data, target_sr


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 via pydub (requires ffmpeg)."""
    from pydub import AudioSegment

    wav_buf = io.BytesIO(wav_bytes)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3")
    return mp3_buf.getvalue()

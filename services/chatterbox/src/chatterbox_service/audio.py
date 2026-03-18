"""Audio format conversion utilities."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import torchaudio

from chatterbox_service.config import settings
from chatterbox_service.models import AudioFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import torch


def wav_to_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform tensor to WAV bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, wav, sample_rate, format="wav")
    return buf.getvalue()


def wav_to_mp3_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform tensor to MP3 bytes via pydub (ffmpeg)."""
    from pydub import AudioSegment

    wav_buf = io.BytesIO()
    torchaudio.save(wav_buf, wav, sample_rate, format="wav")
    wav_buf.seek(0)

    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3")
    return mp3_buf.getvalue()


def encode_audio(wav: torch.Tensor, sample_rate: int, fmt: AudioFormat) -> bytes:
    """Encode waveform to the requested output format."""
    if fmt == AudioFormat.wav:
        return wav_to_bytes(wav, sample_rate)
    if fmt == AudioFormat.mp3:
        return wav_to_mp3_bytes(wav, sample_rate)
    raise ValueError(f"Unsupported format: {fmt}")


async def stream_bytes(data: bytes) -> AsyncIterator[bytes]:
    """Yield *data* in fixed-size chunks for StreamingResponse."""
    chunk_size = settings.stream_chunk_size
    offset = 0
    while offset < len(data):
        yield data[offset : offset + chunk_size]
        offset += chunk_size

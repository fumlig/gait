"""Pydantic schemas matching the OpenAI Audio API contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field
from trave_common import HealthResponse, ModelListResponse, ModelObject

# Re-export shared schemas for backward compatibility
__all__ = [
    "HealthResponse",
    "ModelListResponse",
    "ModelObject",
    "ResponseFormat",
    "Segment",
    "TimestampGranularity",
    "TranscriptionResponse",
    "VerboseTranscriptionResponse",
    "WHISPER_MODEL_SIZES",
    "WordTimestamp",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResponseFormat(StrEnum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TimestampGranularity(StrEnum):
    word = "word"
    segment = "segment"


# Valid whisper model sizes that WhisperX supports.
WHISPER_MODEL_SIZES = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "turbo",
    }
)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class WordTimestamp(BaseModel):
    """A single word with its timestamp."""

    word: str
    start: float
    end: float
    score: float | None = None


class Segment(BaseModel):
    """A transcription segment."""

    id: int = 0
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None


class TranscriptionResponse(BaseModel):
    """JSON response for transcriptions/translations."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """Verbose JSON response with segments and word-level timestamps."""

    task: str = "transcribe"
    language: str = ""
    duration: float = 0.0
    text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)

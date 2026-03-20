"""Pydantic schemas for gateway requests and responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "GatewayHealthResponse",
    "HealthResponse",
    "ModelListResponse",
    "ModelObject",
    "RawSegment",
    "Segment",
    "SpeechRequest",
    "SpeechResponseFormat",
    "TimestampGranularity",
    "TranscriptionResponse",
    "TranscriptionResponseFormat",
    "TranscriptionResult",
    "VerboseTranscriptionResponse",
    "Voice",
    "VoiceListResponse",
    "WordTimestamp",
]


# -- Models --


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = ""
    capabilities: list[str] = Field(default_factory=list)
    loaded: bool = True


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class GatewayHealthResponse(BaseModel):
    status: str = "ok"
    backends: dict[str, str] = {}


# -- Audio enums --


class SpeechResponseFormat(StrEnum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class TranscriptionResponseFormat(StrEnum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TimestampGranularity(StrEnum):
    word = "word"
    segment = "segment"


# -- Speech request --


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body.

    Standard OpenAI fields plus Chatterbox-specific extensions.
    """

    model: str
    input: str = Field(..., max_length=4096)
    voice: str
    response_format: SpeechResponseFormat = SpeechResponseFormat.mp3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Chatterbox-specific extensions
    language: str | None = None
    exaggeration: float | None = Field(default=None, ge=0.0, le=2.0)
    cfg_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=3.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None


# -- Transcription / Translation --


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    score: float | None = None


class Segment(BaseModel):
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
    text: str


class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str = ""
    duration: float = 0.0
    text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)


class RawSegment(BaseModel):
    """A raw segment from the STT backend, before formatting."""

    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    """Internal result from the STT backend, before response formatting."""

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[RawSegment] = Field(default_factory=list)


# -- Voice management --


class Voice(BaseModel):
    voice_id: str
    name: str


class VoiceListResponse(BaseModel):
    object: str = "list"
    data: list[Voice] = Field(default_factory=list)

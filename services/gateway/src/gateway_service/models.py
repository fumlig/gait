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


# ---------------------------------------------------------------------------
# Shared schemas (inlined from trave-common)
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    """A single model entry in the OpenAI /v1/models response."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = ""


class ModelListResponse(BaseModel):
    """GET /v1/models response body."""

    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health response body for GPU-backed services."""

    status: str = "ok"
    model_loaded: bool = False


# ---------------------------------------------------------------------------
# Gateway health
# ---------------------------------------------------------------------------


class GatewayHealthResponse(BaseModel):
    """Gateway-specific health response with backend status."""

    status: str = "ok"
    backends: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Audio enums
# ---------------------------------------------------------------------------


class SpeechResponseFormat(StrEnum):
    """Output formats for POST /v1/audio/speech."""

    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class TranscriptionResponseFormat(StrEnum):
    """Output formats for POST /v1/audio/transcriptions and /translations."""

    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TimestampGranularity(StrEnum):
    """Timestamp granularity options for transcription."""

    word = "word"
    segment = "segment"


# ---------------------------------------------------------------------------
# Speech request
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body (OpenAI-compatible).

    Standard OpenAI fields: model, input, voice, response_format, speed.
    Extended fields for Chatterbox-specific controls: language, exaggeration,
    cfg_weight, temperature, repetition_penalty, top_p, min_p, top_k, seed.
    """

    model: str = Field(
        ...,
        description="Model ID to use for speech generation.",
    )
    input: str = Field(
        ...,
        max_length=4096,
        description="The text to generate audio for.",
    )
    voice: str = Field(
        ...,
        description="Voice name or identifier.",
    )
    response_format: SpeechResponseFormat = Field(
        default=SpeechResponseFormat.mp3,
        description="Audio output format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier.",
    )

    # -- Extended fields (Chatterbox-specific, passed through to backend) --

    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'fr'). Required for multilingual models.",
    )
    exaggeration: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Emotion exaggeration level.",
    )
    cfg_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance weight controlling pace.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.01,
        le=5.0,
        description="Sampling temperature for generation.",
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=1.0,
        le=3.0,
        description="Token repetition penalty.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
    min_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min-p sampling threshold.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility.",
    )


# ---------------------------------------------------------------------------
# Transcription / Translation responses
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


# ---------------------------------------------------------------------------
# Internal transcription result (returned by TranscriptionClient)
# ---------------------------------------------------------------------------


class RawSegment(BaseModel):
    """A raw segment from the STT backend (minimal, pre-formatting)."""

    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    """Structured result from the STT backend, before response formatting.

    This is the internal contract between the gateway and transcription
    clients.  The gateway handles formatting into the OpenAI-compatible
    response shapes (json, text, srt, vtt, verbose_json).
    """

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[RawSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------


class Voice(BaseModel):
    """A voice reference clip available for TTS."""

    voice_id: str
    name: str


class VoiceListResponse(BaseModel):
    """Wrapped list response for voices (OpenAI list pattern)."""

    object: str = "list"
    data: list[Voice] = Field(default_factory=list)

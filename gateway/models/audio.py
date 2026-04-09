"""Audio: speech, transcription, translation, and voice management.

Some upstream providers omit fields that the OpenAI specification marks
as always-present in audio transcription streaming events.  The typed
event models in this module declare all spec-required fields with
sensible defaults so that ``model_validate`` transparently fills them
in, just like the Responses API event models in ``responses.py``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Speech request
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body.

    Standard OpenAI fields plus Chatterbox-specific extensions.
    """

    model: str = Field(..., min_length=1)
    input: str = Field(..., max_length=4096)
    voice: str
    response_format: SpeechResponseFormat = SpeechResponseFormat.mp3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Chatterbox extensions (not part of OpenAI API)
    language: str | None = None
    exaggeration: float | None = Field(default=None, ge=0.0, le=2.0)
    cfg_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=3.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None


# ---------------------------------------------------------------------------
# Transcription / translation
# ---------------------------------------------------------------------------


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
    """A raw segment from the STT backend, before formatting.

    Internal gateway type — not part of the OpenAI API.
    """

    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None  # whisperx diarization extension


class TranscriptionResult(BaseModel):
    """Internal result from the STT backend, before response formatting.

    Internal gateway type — not part of the OpenAI API.
    """

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[RawSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Transcription streaming — event models
#
# The OpenAI audio transcription streaming API emits typed SSE events.
# Each model uses a Literal ``type`` discriminator and is combined into
# the ``TranscriptionStreamEvent`` union, mirroring the pattern in
# ``responses.py``.
# ---------------------------------------------------------------------------


class TranscriptionLogprob(BaseModel):
    """A single token log-probability entry."""

    model_config = ConfigDict(extra="allow")

    token: str = ""
    logprob: float = 0.0
    bytes: list[int] | None = None


class TranscriptionCreatedEvent(BaseModel):
    """``transcription.created`` — initial lifecycle event."""

    model_config = ConfigDict(extra="allow")

    type: Literal["transcription.created"] = "transcription.created"


class TranscriptionTextDeltaEvent(BaseModel):
    """``transcription.text.delta`` — incremental transcript text."""

    model_config = ConfigDict(extra="allow")

    type: Literal["transcription.text.delta"] = "transcription.text.delta"
    delta: str = ""
    logprobs: list[TranscriptionLogprob] = Field(default_factory=list)


class TranscriptionTextDoneEvent(BaseModel):
    """``transcription.text.done`` — accumulated full transcript text."""

    model_config = ConfigDict(extra="allow")

    type: Literal["transcription.text.done"] = "transcription.text.done"
    text: str = ""
    logprobs: list[TranscriptionLogprob] = Field(default_factory=list)


class TranscriptionCompletedEvent(BaseModel):
    """``transcription.completed`` — final lifecycle event."""

    model_config = ConfigDict(extra="allow")

    type: Literal["transcription.completed"] = "transcription.completed"


def _get_transcription_event_discriminator(v: Any) -> str:
    """Extract the ``type`` tag for discriminated-union dispatch."""
    if isinstance(v, dict):
        return v.get("type", "")
    return getattr(v, "type", "")


TranscriptionStreamEvent = Annotated[
    Annotated[TranscriptionCreatedEvent, Tag("transcription.created")]
    | Annotated[TranscriptionTextDeltaEvent, Tag("transcription.text.delta")]
    | Annotated[TranscriptionTextDoneEvent, Tag("transcription.text.done")]
    | Annotated[TranscriptionCompletedEvent, Tag("transcription.completed")],
    Discriminator(_get_transcription_event_discriminator),
]
"""Discriminated union of audio transcription streaming event types."""


# ---------------------------------------------------------------------------
# Voice management (gateway extension — not part of the OpenAI API)
# ---------------------------------------------------------------------------


class Voice(BaseModel):
    voice_id: str
    name: str


class VoiceListResponse(BaseModel):
    object: str = "list"
    data: list[Voice] = Field(default_factory=list)

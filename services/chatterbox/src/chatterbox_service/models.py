"""Pydantic schemas matching the OpenAI API contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AudioFormat(StrEnum):
    mp3 = "mp3"
    wav = "wav"


CONTENT_TYPE_MAP: dict[AudioFormat, str] = {
    AudioFormat.mp3: "audio/mpeg",
    AudioFormat.wav: "audio/wav",
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body (OpenAI-compatible)."""

    model: str = Field(
        ...,
        description="Model ID. Use 'chatterbox-turbo'.",
    )
    input: str = Field(
        ...,
        max_length=4096,
        description="The text to generate audio for.",
    )
    voice: str = Field(
        ...,
        description="Voice name corresponding to a registered reference clip.",
    )
    response_format: AudioFormat = Field(
        default=AudioFormat.mp3,
        description="Audio output format: mp3 or wav.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "resemble-ai"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False

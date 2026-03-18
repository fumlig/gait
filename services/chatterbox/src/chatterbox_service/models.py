"""Pydantic schemas matching the OpenAI API contract."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field
from trave_common import HealthResponse as _BaseHealthResponse
from trave_common import ModelListResponse, ModelObject

# Re-export shared schemas for backward compatibility
__all__ = [
    "CONTENT_TYPE_MAP",
    "SUPPORTED_LANGUAGES",
    "AudioFormat",
    "HealthResponse",
    "ModelListResponse",
    "ModelObject",
    "SpeechRequest",
]


# ---------------------------------------------------------------------------
# Health (extended with per-model status)
# ---------------------------------------------------------------------------


class HealthResponse(_BaseHealthResponse):
    """Chatterbox health response with per-model loaded status."""

    models: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-model loaded status (model_name -> is_loaded).",
    )


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
# Language support
# ---------------------------------------------------------------------------


SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body (OpenAI-compatible).

    Standard OpenAI fields: model, input, voice, response_format, speed.
    Extended fields for Chatterbox-specific controls: language, exaggeration,
    cfg_weight, temperature, repetition_penalty, top_p, min_p, top_k, seed.
    """

    model: str = Field(
        ...,
        description="Model ID: 'chatterbox-turbo', 'chatterbox', or 'chatterbox-multilingual'.",
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

    # -- Extended fields (Chatterbox-specific) --

    language: str | None = Field(
        default=None,
        description=(
            "ISO 639-1 language code (e.g. 'en', 'fr', 'zh'). "
            "Required for chatterbox-multilingual, ignored by other models."
        ),
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description=(
            "Emotion exaggeration level. 0.5 = neutral, higher = more expressive. "
            "Used by chatterbox and chatterbox-multilingual; ignored by chatterbox-turbo."
        ),
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier-free guidance weight controlling pace. "
            "Used by chatterbox and chatterbox-multilingual; ignored by chatterbox-turbo."
        ),
    )
    temperature: float = Field(
        default=0.8,
        ge=0.01,
        le=5.0,
        description="Sampling temperature for generation.",
    )
    repetition_penalty: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="Token repetition penalty.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
    min_p: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Min-p sampling threshold (0 to disable). Used by chatterbox and multilingual."
        ),
    )
    top_k: int = Field(
        default=1000,
        ge=1,
        description="Top-k sampling (used by chatterbox-turbo).",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. None or 0 for random.",
    )

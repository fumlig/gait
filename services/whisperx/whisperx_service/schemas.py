"""Pydantic schemas for the whisperx service HTTP API.

The file-upload endpoints (``/transcribe``, ``/translate``,
``/transcribe_stream``) keep their multipart bodies, but their
non-file form fields are validated through ``TranscribeFormRequest``
(built from the form dict).  Model-management endpoints
(``/models/load``, ``/models/unload``) use plain JSON bodies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StatusPhase = Literal["unloaded", "loading", "loaded", "sleeping"]


# ---------------------------------------------------------------------------
# Transcribe / translate form payload
# ---------------------------------------------------------------------------


class TranscribeFormRequest(BaseModel):
    """Validated form fields for ``/transcribe`` and ``/translate``.

    The uploaded ``file`` is handled separately (Starlette UploadFile);
    this model only validates the textual form fields.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: str = Field(..., min_length=1)
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    word_timestamps: bool = False
    diarize: bool = False


class TranscribeStreamFormRequest(BaseModel):
    """Validated form fields for ``/transcribe_stream``."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: str = Field(..., min_length=1)
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


class ModelStatus(BaseModel):
    """Lifecycle status of a model, mirroring llama-server's shape."""

    model_config = ConfigDict(extra="allow")

    value: StatusPhase = "unloaded"
    args: list[str] = Field(default_factory=list)
    failed: bool = False
    exit_code: int | None = None


class ModelInfo(BaseModel):
    """Single entry in the ``/models`` response."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "whisperx"
    capabilities: list[str] = Field(
        default_factory=lambda: ["transcription", "translation"],
    )
    status: ModelStatus = Field(default_factory=ModelStatus)
    loaded: bool = False


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model: str = Field(..., min_length=1)


class UnloadModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model: str = Field(..., min_length=1)


class LoadModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool
    model: str
    status: ModelStatus


class UnloadModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool
    model: str
    status: ModelStatus


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool = False
    loaded_model: str | None = None
    phase: StatusPhase = "unloaded"

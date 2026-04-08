"""Pydantic schemas for the chatterbox service HTTP API.

Starlette handlers validate request bodies through these models and
serialise responses via ``model_dump``.  The schemas mirror the
gateway's shapes where possible so the gait gateway can forward them
unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StatusPhase = Literal["unloaded", "loading", "loaded", "sleeping"]


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """Body of ``POST /synthesize``.

    Required: ``text``, ``voice``, ``model``.  Everything else is a
    tunable with a per-model default (the engine applies defaults
    internally).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    text: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)

    language: str | None = None
    speed: float = 1.0
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    top_p: float = 1.0
    min_p: float = 0.05
    top_k: int = 1000
    seed: int | None = None


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
    owned_by: str = "resemble-ai"
    capabilities: list[str] = Field(default_factory=lambda: ["speech"])
    status: ModelStatus = Field(default_factory=ModelStatus)
    # Legacy flag kept for backwards compatibility with the old
    # ``/models`` response shape. Derived from ``status.value``.
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
    models: dict[str, bool] = Field(default_factory=dict)
    phase: StatusPhase = "unloaded"

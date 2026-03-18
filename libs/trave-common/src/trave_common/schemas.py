"""OpenAI-compatible response schemas shared across all trave services."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelObject(BaseModel):
    """A single model entry in the OpenAI /v1/models response."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = ""


class ModelListResponse(BaseModel):
    """GET /v1/models response body."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    """GET /health response body for GPU-backed services."""

    status: str = "ok"
    model_loaded: bool = False

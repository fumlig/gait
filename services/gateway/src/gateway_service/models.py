"""Pydantic schemas for gateway responses."""

from __future__ import annotations

from pydantic import BaseModel
from trave_common import HealthResponse, ModelListResponse, ModelObject

# Re-export shared schemas for backward compatibility
__all__ = [
    "BackendHealth",
    "GatewayHealthResponse",
    "HealthResponse",
    "ModelListResponse",
    "ModelObject",
]


class GatewayHealthResponse(BaseModel):
    """Gateway-specific health response with backend status."""

    status: str = "ok"
    backends: dict[str, str] = {}


class BackendHealth(BaseModel):
    status: str = "ok"
    model_loaded: bool = False

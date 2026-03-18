"""Pydantic schemas for gateway responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = ""


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    status: str = "ok"
    backends: dict[str, str] = {}


class BackendHealth(BaseModel):
    status: str = "ok"
    model_loaded: bool = False

"""Shared types: health, model listing, usage, response format, stream options."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Health & model listing
# ---------------------------------------------------------------------------


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
    backends: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared usage / options
# ---------------------------------------------------------------------------


class CompletionUsage(BaseModel):
    """Token usage for chat completions and text completions."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseFormat(BaseModel):
    """Requested response format (text, json_object, json_schema)."""

    model_config = ConfigDict(extra="allow")

    type: str  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False

"""Shared types: health, model listing, usage, response format, stream options."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Health & model listing
# ---------------------------------------------------------------------------

ModelStatusValue = Literal["unloaded", "loading", "loaded", "sleeping"]


class ModelStatus(BaseModel):
    """Lifecycle status of a model, mirroring llama-server's shape.

    ``value`` is one of:

    - ``unloaded`` — never loaded or explicitly unloaded
    - ``loading``  — currently being loaded
    - ``loaded``   — resident and ready to serve requests
    - ``sleeping`` — auto-unloaded due to idle timeout; will reload lazily
    """

    model_config = ConfigDict(extra="allow")

    value: ModelStatusValue = "unloaded"
    args: list[str] = Field(default_factory=list)
    failed: bool = False
    exit_code: int | None = None


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = ""
    capabilities: list[str] = Field(default_factory=list)  # gateway extension
    status: ModelStatus | None = None  # gateway extension
    loaded: bool = True  # gateway extension (derived from status when present)

    @model_validator(mode="after")
    def _sync_loaded_flag(self) -> ModelObject:
        """Keep ``loaded`` consistent with ``status.value`` when both set."""
        if self.status is not None:
            self.loaded = self.status.value == "loaded"
        return self


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model management (gait extension — not in the OpenAI API)
# ---------------------------------------------------------------------------


class LoadModelRequest(BaseModel):
    """Request body for ``POST /v1/models/load``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = Field(..., min_length=1, description="Model id to load.")


class LoadModelResponse(BaseModel):
    """Response body for ``POST /v1/models/load``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    success: bool
    model: str
    status: ModelStatus | None = None


class UnloadModelRequest(BaseModel):
    """Request body for ``POST /v1/models/unload``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = Field(..., min_length=1, description="Model id to unload.")


class UnloadModelResponse(BaseModel):
    """Response body for ``POST /v1/models/unload``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    success: bool
    model: str
    status: ModelStatus | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class GatewayHealthResponse(BaseModel):
    """Gateway extension — not part of the OpenAI API."""

    status: str = "ok"
    backends: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared usage / options
# ---------------------------------------------------------------------------


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion token counts."""

    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt token counts."""

    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0


class CompletionUsage(BaseModel):
    """Token usage for chat completions and text completions."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ResponseFormat(BaseModel):
    """Requested response format (text, json_object, json_schema)."""

    model_config = ConfigDict(extra="allow")

    type: str  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False

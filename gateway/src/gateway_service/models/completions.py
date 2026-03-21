"""Text completions: request and response."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway_service.models.common import (  # noqa: TC001 — Pydantic resolves these at runtime
    CompletionUsage,
    StreamOptions,
)

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """POST /v1/completions request body (OpenAI-compatible, legacy)."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[Any] = Field(
        ...,
        description=(
            "Prompt(s) to complete. A string, array of strings, "
            "array of token IDs, or array of token-ID arrays."
        ),
    )
    best_of: int | None = None
    echo: bool | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    stream_options: StreamOptions | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    text: str
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class CompletionResponse(BaseModel):
    """Response from POST /v1/completions (non-streaming)."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None

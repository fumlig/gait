"""Embeddings: request and response."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """POST /v1/embeddings request body (OpenAI-compatible)."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1)
    input: str | list[Any] = Field(
        ...,
        description=(
            "Input text(s) to embed. A string, array of strings, "
            "array of token IDs, or array of token-ID arrays."
        ),
    )
    encoding_format: str | None = None  # "float" | "base64"
    dimensions: int | None = None
    user: str | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class EmbeddingObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str  # float array or base64 string


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response from POST /v1/embeddings."""

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage | None = None

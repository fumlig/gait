"""Responses API: request and response."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CreateResponseRequest(BaseModel):
    """POST /v1/responses request body (OpenAI Responses API)."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]] = Field(
        ...,
        description="A text string or array of input items (messages, etc.).",
    )
    instructions: str | None = None
    # Tool calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Sampling
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    # Streaming
    stream: bool = False
    # Metadata
    metadata: dict[str, str] | None = None
    store: bool | None = None
    previous_response_id: str | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ResponseOutputContent(BaseModel):
    """A content block in a response output message."""

    model_config = ConfigDict(extra="allow")

    type: str  # "output_text", "refusal", etc.
    text: str | None = None


class ResponseOutputMessage(BaseModel):
    """A message in the response output array."""

    model_config = ConfigDict(extra="allow")

    type: str = "message"
    id: str = ""
    role: str = "assistant"
    content: list[ResponseOutputContent] = Field(default_factory=list)
    status: str | None = None


class ResponseUsage(BaseModel):
    """Token usage for the Responses API."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class CreateResponseResponse(BaseModel):
    """Response from POST /v1/responses (non-streaming)."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: int = 0
    model: str
    output: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"
    usage: ResponseUsage | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    incomplete_details: dict[str, Any] | None = None

"""Chat completions: request, response, streaming chunks, tool calling."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway_service.models.common import (  # noqa: TC001 — Pydantic resolves these at runtime
    CompletionUsage,
    ResponseFormat,
    StreamOptions,
)

# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class FunctionDefinition(BaseModel):
    """Function schema within a tool definition."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ChatCompletionTool(BaseModel):
    """A tool the model may call."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function invocation made by the model."""

    name: str
    arguments: str


class ChatCompletionToolCall(BaseModel):
    """A tool call inside an assistant message."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Covers all roles: system, developer, user, assistant, tool.
    Role-specific fields are optional — the backend validates constraints.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    # assistant messages
    tool_calls: list[ChatCompletionToolCall] | None = None
    refusal: str | None = None
    # tool messages
    tool_call_id: str | None = None


class ChatAudioConfig(BaseModel):
    """Audio output configuration for chat completions."""

    model_config = ConfigDict(extra="allow")

    voice: str = "default"
    model: str | None = None
    format: str | None = None  # "pcm16", "wav", etc.


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body (OpenAI-compatible)."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    # Tool calling
    tools: list[ChatCompletionTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Response format
    response_format: ResponseFormat | None = None
    seed: int | None = None
    # Audio modality (gateway extension)
    modalities: list[str] | None = None
    audio: ChatAudioConfig | None = None
    # Streaming options
    stream_options: StreamOptions | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class TopLogprob(BaseModel):
    model_config = ConfigDict(extra="allow")

    token: str
    logprob: float
    bytes: list[int] | None = None


class ChatCompletionTokenLogprob(BaseModel):
    model_config = ConfigDict(extra="allow")

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: list[ChatCompletionTokenLogprob] | None = None
    refusal: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    """Response from POST /v1/chat/completions (non-streaming)."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Streaming chunks
# ---------------------------------------------------------------------------


class ChatCompletionChunkDelta(BaseModel):
    """Delta payload in a streaming chat completion chunk."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    refusal: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    """A single server-sent event in a streaming chat completion."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None

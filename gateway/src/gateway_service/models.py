"""Pydantic schemas for gateway requests and responses.

Covers all OpenAI-compatible endpoints: chat completions, completions,
embeddings, responses, audio speech/transcription/translation, and voices.
Models use ``extra="allow"`` so unknown fields from backends pass through.
"""

from __future__ import annotations

from enum import StrEnum
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
# Shared types — usage
# ---------------------------------------------------------------------------


class CompletionUsage(BaseModel):
    """Token usage for chat completions and text completions."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Shared types — tool calling
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
# Shared types — response format / stream options
# ---------------------------------------------------------------------------


class ResponseFormat(BaseModel):
    """Requested response format (text, json_object, json_schema)."""

    model_config = ConfigDict(extra="allow")

    type: str  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


# ---------------------------------------------------------------------------
# Chat completions — request
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
# Chat completions — response
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
# Chat completions — streaming chunks
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


# ---------------------------------------------------------------------------
# Text completions — request
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
# Text completions — response
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


# ---------------------------------------------------------------------------
# Embeddings — request
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """POST /v1/embeddings request body (OpenAI-compatible)."""

    model_config = ConfigDict(extra="allow")

    model: str
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
# Embeddings — response
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


# ---------------------------------------------------------------------------
# Responses API — request
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
# Responses API — response
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


# ---------------------------------------------------------------------------
# Audio enums
# ---------------------------------------------------------------------------


class SpeechResponseFormat(StrEnum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class TranscriptionResponseFormat(StrEnum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TimestampGranularity(StrEnum):
    word = "word"
    segment = "segment"


# ---------------------------------------------------------------------------
# Audio — speech request
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body.

    Standard OpenAI fields plus Chatterbox-specific extensions.
    """

    model: str
    input: str = Field(..., max_length=4096)
    voice: str
    response_format: SpeechResponseFormat = SpeechResponseFormat.mp3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Chatterbox-specific extensions
    language: str | None = None
    exaggeration: float | None = Field(default=None, ge=0.0, le=2.0)
    cfg_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=3.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None


# ---------------------------------------------------------------------------
# Audio — transcription / translation
# ---------------------------------------------------------------------------


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    score: float | None = None


class Segment(BaseModel):
    id: int = 0
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None


class TranscriptionResponse(BaseModel):
    text: str


class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str = ""
    duration: float = 0.0
    text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)


class RawSegment(BaseModel):
    """A raw segment from the STT backend, before formatting."""

    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    """Internal result from the STT backend, before response formatting."""

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[RawSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------


class Voice(BaseModel):
    voice_id: str
    name: str


class VoiceListResponse(BaseModel):
    object: str = "list"
    data: list[Voice] = Field(default_factory=list)

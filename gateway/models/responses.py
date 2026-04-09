"""Responses API: request, response, and streaming event models.

Some upstream providers (notably llama.cpp) omit fields that the OpenAI
specification marks as always-present in Responses API payloads, or emit
them as an explicit ``null``.  Strict clients that validate against the
spec fail in both cases.

To work around this, every model in this module declares **all**
spec-required fields with sensible defaults.  When an upstream response
is validated through these models (via ``model_validate`` or the
``ResponseStreamEvent`` discriminated union), any omitted fields are
transparently filled in so the gateway's output always satisfies the
OpenAI contract — regardless of how complete the backend's response is.

For spec-required fields whose type is a nested struct (not
``Optional``), we additionally attach a ``BeforeValidator`` that coerces
an incoming ``None`` to an empty dict, so the field's ``default_factory``
produces the default instance instead of raising a validation error.
See ``_none_as_empty`` and its uses in ``ResponseUsage``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, Tag


def _none_as_empty(v: Any) -> Any:
    """Coerce an explicit ``None`` to ``{}`` for spec-required struct fields.

    llama.cpp sometimes emits ``"output_tokens_details": null`` (and
    similar) even though the OpenAI Responses API spec marks these
    fields as always-present non-null structs.  Returning ``{}`` here
    lets Pydantic fall through to the field's ``default_factory`` /
    declared defaults, so the serialized gateway response always
    contains a valid struct instead of ``null``.
    """
    return {} if v is None else v

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ReasoningConfig(BaseModel):
    """Reasoning configuration for the Responses API."""

    model_config = ConfigDict(extra="allow")

    effort: str | None = None  # "low", "medium", "high"
    summary: str | None = None  # "auto", "concise", "detailed"


class CreateResponseRequest(BaseModel):
    """POST /v1/responses request body (OpenAI Responses API)."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1)
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
    top_k: int | None = None
    min_p: float | None = None
    max_output_tokens: int | None = None
    # Reasoning
    reasoning: ReasoningConfig | None = None
    # Streaming
    stream: bool = False
    # Metadata
    metadata: dict[str, str] | None = None
    store: bool | None = None
    previous_response_id: str | None = None


# ---------------------------------------------------------------------------
# Response — inner models
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


class ReasoningSummaryContent(BaseModel):
    """A content block in a reasoning summary."""

    model_config = ConfigDict(extra="allow")

    type: str = "summary_text"
    text: str = ""


class ResponseReasoningItem(BaseModel):
    """A reasoning item in the response output array."""

    model_config = ConfigDict(extra="allow")

    type: str = "reasoning"
    id: str = ""
    summary: list[ReasoningSummaryContent] = Field(default_factory=list)


class OutputTokensDetails(BaseModel):
    """Breakdown of output token counts for the Responses API."""

    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int = 0


class InputTokensDetails(BaseModel):
    """Breakdown of input token counts for the Responses API."""

    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0


class ResponseUsage(BaseModel):
    """Token usage for the Responses API.

    ``output_tokens_details`` and ``input_tokens_details`` are declared
    as always-present non-null structs by the OpenAI spec.  We use
    ``BeforeValidator(_none_as_empty)`` so that an explicit ``null``
    from the upstream provider is coerced to the default instance
    (``{"reasoning_tokens": 0}`` / ``{"cached_tokens": 0}``) rather
    than propagated as ``null`` to strict clients.
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    output_tokens_details: Annotated[
        OutputTokensDetails, BeforeValidator(_none_as_empty),
    ] = Field(default_factory=OutputTokensDetails)
    input_tokens_details: Annotated[
        InputTokensDetails, BeforeValidator(_none_as_empty),
    ] = Field(default_factory=InputTokensDetails)


class ResponseTextFormat(BaseModel):
    """Format specifier within the text response configuration."""

    model_config = ConfigDict(extra="allow")

    type: str = "text"


class ResponseTextConfig(BaseModel):
    """Text response format configuration on the Response object."""

    model_config = ConfigDict(extra="allow")

    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)


def _now_ts() -> int:
    """Return the current Unix timestamp (seconds)."""
    return int(time.time())


# ---------------------------------------------------------------------------
# Response — top-level
# ---------------------------------------------------------------------------


class CreateResponseResponse(BaseModel):
    """Response from POST /v1/responses.

    All fields that the OpenAI specification marks as always-present are
    declared here with sensible defaults so that ``model_validate()``
    fills them in when an upstream provider omits them.

    Defaults are chosen to match what the OpenAI API itself would return
    for a typical request:

    * ``created_at`` — current Unix timestamp (gateway receive time).
    * ``model`` — empty string.  llama.cpp's ``response.created`` and
      ``response.in_progress`` lifecycle SSE events nest a truncated
      ``response`` object that omits ``model``; this default lets those
      events validate instead of being silently dropped.
    * ``temperature`` / ``top_p`` — ``1.0`` (OpenAI sampling defaults).
    * ``tool_choice`` — ``"auto"`` (OpenAI default).
    * ``metadata`` — empty dict (OpenAI sends ``{}``).
    * ``text`` — ``{"format": {"type": "text"}}`` (plain-text format).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=_now_ts)
    model: str = ""
    output: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"
    # Token usage
    usage: ResponseUsage | None = None
    # Error / incomplete
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    # Configuration echoed back
    instructions: str | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    reasoning: ReasoningConfig | None = None
    store: bool = True
    temperature: float | None = 1.0
    text: ResponseTextConfig = Field(default_factory=ResponseTextConfig)
    tool_choice: str | dict[str, Any] = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_p: float | None = 1.0
    truncation: str = "disabled"
    user: str | None = None


# ---------------------------------------------------------------------------
# Streaming — inner models
# ---------------------------------------------------------------------------


class OutputItem(BaseModel):
    """An output item in a streaming event (message, function_call, etc.).

    Common fields are declared with defaults so that clients always see
    them, even when the upstream provider omits them.  Type-specific
    fields (``role``, ``content``, ``arguments``, …) are preserved
    via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    id: str = ""
    status: str = "in_progress"


class ContentPart(BaseModel):
    """A content part within an output message (streaming events).

    The ``text`` and ``annotations`` fields are always present in the
    OpenAI spec for ``output_text`` parts; we default them so that
    providers that omit them don't break clients.  Extra fields from
    the upstream server are preserved.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming — event wrappers (one model per OpenAI event type)
#
# Each model uses a Literal ``type`` discriminator so they can be
# combined into a single ``ResponseStreamEvent`` discriminated union.
# Pydantic dispatches on the ``type`` value and fills any missing
# fields with their spec-conformant defaults.
# ---------------------------------------------------------------------------

# -- Response lifecycle events ------------------------------------------------


class _ResponseEventBase(BaseModel):
    """Shared fields for response-level lifecycle events."""

    model_config = ConfigDict(extra="allow")

    sequence_number: int = 0
    response: CreateResponseResponse


class ResponseCreatedEvent(_ResponseEventBase):
    type: Literal["response.created"] = "response.created"


class ResponseInProgressEvent(_ResponseEventBase):
    type: Literal["response.in_progress"] = "response.in_progress"


class ResponseCompletedEvent(_ResponseEventBase):
    type: Literal["response.completed"] = "response.completed"


class ResponseFailedEvent(_ResponseEventBase):
    type: Literal["response.failed"] = "response.failed"


class ResponseIncompleteEvent(_ResponseEventBase):
    type: Literal["response.incomplete"] = "response.incomplete"


# -- Output-item lifecycle events ---------------------------------------------


class _OutputItemEventBase(BaseModel):
    """Shared fields for output-item lifecycle events."""

    model_config = ConfigDict(extra="allow")

    sequence_number: int = 0
    output_index: int = 0
    item: OutputItem


class OutputItemAddedEvent(_OutputItemEventBase):
    type: Literal["response.output_item.added"] = "response.output_item.added"


class OutputItemDoneEvent(_OutputItemEventBase):
    type: Literal["response.output_item.done"] = "response.output_item.done"


# -- Content-part lifecycle events --------------------------------------------


class _ContentPartEventBase(BaseModel):
    """Shared fields for content-part lifecycle events."""

    model_config = ConfigDict(extra="allow")

    sequence_number: int = 0
    output_index: int = 0
    content_index: int = 0
    part: ContentPart


class ContentPartAddedEvent(_ContentPartEventBase):
    type: Literal["response.content_part.added"] = "response.content_part.added"


class ContentPartDoneEvent(_ContentPartEventBase):
    type: Literal["response.content_part.done"] = "response.content_part.done"


# -- Text delta / done events ------------------------------------------------


class OutputTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    sequence_number: int = 0
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class OutputTextDoneEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.output_text.done"] = "response.output_text.done"
    sequence_number: int = 0
    output_index: int = 0
    content_index: int = 0
    text: str = ""


# -- Refusal delta / done events ----------------------------------------------


class RefusalDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    sequence_number: int = 0
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class RefusalDoneEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.refusal.done"] = "response.refusal.done"
    sequence_number: int = 0
    output_index: int = 0
    content_index: int = 0
    refusal: str = ""


# -- Function-call argument delta / done events -------------------------------


class FuncCallArgsDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    sequence_number: int = 0
    output_index: int = 0
    item_id: str = ""
    delta: str = ""


class FuncCallArgsDoneEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    sequence_number: int = 0
    output_index: int = 0
    item_id: str = ""
    arguments: str = ""


# ---------------------------------------------------------------------------
# Discriminated union — single entry point for validating any stream event
# ---------------------------------------------------------------------------


def _get_stream_event_discriminator(v: Any) -> str:
    """Extract the ``type`` tag for discriminated-union dispatch.

    Works for both raw dicts (before validation) and model instances.
    """
    if isinstance(v, dict):
        return v.get("type", "")
    return getattr(v, "type", "")


ResponseStreamEvent = Annotated[
    Annotated[ResponseCreatedEvent, Tag("response.created")]
    | Annotated[ResponseInProgressEvent, Tag("response.in_progress")]
    | Annotated[ResponseCompletedEvent, Tag("response.completed")]
    | Annotated[ResponseFailedEvent, Tag("response.failed")]
    | Annotated[ResponseIncompleteEvent, Tag("response.incomplete")]
    | Annotated[OutputItemAddedEvent, Tag("response.output_item.added")]
    | Annotated[OutputItemDoneEvent, Tag("response.output_item.done")]
    | Annotated[ContentPartAddedEvent, Tag("response.content_part.added")]
    | Annotated[ContentPartDoneEvent, Tag("response.content_part.done")]
    | Annotated[OutputTextDeltaEvent, Tag("response.output_text.delta")]
    | Annotated[OutputTextDoneEvent, Tag("response.output_text.done")]
    | Annotated[RefusalDeltaEvent, Tag("response.refusal.delta")]
    | Annotated[RefusalDoneEvent, Tag("response.refusal.done")]
    | Annotated[FuncCallArgsDeltaEvent, Tag("response.function_call_arguments.delta")]
    | Annotated[FuncCallArgsDoneEvent, Tag("response.function_call_arguments.done")],
    Discriminator(_get_stream_event_discriminator),
]
"""Discriminated union of all Responses API streaming event types.

Pass a raw ``dict`` to
``TypeAdapter(ResponseStreamEvent).validate_python()`` and Pydantic will
select the correct model based on the ``type`` field, filling in any
missing fields with spec-conformant defaults.
"""

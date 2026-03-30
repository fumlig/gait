"""Llama.cpp provider client — typed proxy to the llama.cpp server.

Each method maps an OpenAI endpoint to the corresponding backend path
and response model.  The actual HTTP transport is handled by the
``forward``, ``forward_stream``, and ``stream_raw`` helpers from
``providers.transport``.

For the Responses API stream, each SSE event is parsed from the
upstream byte stream, validated through the ``ResponseStreamEvent``
discriminated union (filling spec-required defaults), and yielded as
a typed Pydantic model.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar

from pydantic import TypeAdapter, ValidationError

from gateway.models import (
    ChatCompletionResponse,
    CompletionResponse,
    CreateResponseResponse,
    EmbeddingResponse,
)
from gateway.models.responses import ResponseStreamEvent
from gateway.providers.base import BaseProvider
from gateway.providers.protocols import (
    ChatCompletions,
    Completions,
    Embeddings,
    Responses,
)
from gateway.providers.transport import forward, forward_stream, stream_raw

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from starlette.responses import StreamingResponse

    from gateway.models import (
        ChatCompletionRequest,
        CompletionRequest,
        CreateResponseRequest,
        EmbeddingRequest,
    )

logger = logging.getLogger(__name__)

_stream_event_adapter = TypeAdapter(ResponseStreamEvent)


# ---------------------------------------------------------------------------
# Responses-stream SSE → typed event parsing
# ---------------------------------------------------------------------------


def _parse_sse_event(
    block: bytes,
    seq: int,
) -> tuple[ResponseStreamEvent | None, int]:
    """Parse one SSE block into a validated event model.

    Returns ``(event, next_seq)``.  *event* is ``None`` when the block
    is a ``[DONE]`` sentinel, a keep-alive comment, or something we
    cannot validate (unknown type, bad JSON, etc.) — the caller decides
    how to handle those.
    """
    lines = block.split(b"\n")

    # Locate the single ``data:`` line.
    data_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(b"data:"):
            if data_idx is not None:
                return None, seq  # multiple data lines → skip
            data_idx = i

    if data_idx is None:
        return None, seq  # no data line (comment / keep-alive)

    raw = lines[data_idx][5:]  # strip "data:"
    if raw.startswith(b" "):
        raw = raw[1:]  # strip optional leading space

    if raw.strip() == b"[DONE]":
        return None, seq

    try:
        obj: dict = json.loads(raw)
    except json.JSONDecodeError:
        return None, seq

    if "type" not in obj:
        return None, seq

    had_seq = "sequence_number" in obj

    try:
        event = _stream_event_adapter.validate_python(obj)
    except ValidationError:
        return None, seq  # unknown or malformed event type

    if not had_seq:
        event.sequence_number = seq
    seq += 1

    return event, seq


async def _parse_response_stream(
    resp: httpx.Response,
) -> AsyncIterator[ResponseStreamEvent]:
    """Yield typed event models from an upstream SSE byte stream.

    Each ``data:`` payload is validated through the
    ``ResponseStreamEvent`` discriminated union so that missing fields
    are filled with spec-conformant defaults.  Non-event lines (comments,
    ``[DONE]``) are silently dropped — the route adds its own ``[DONE]``
    after the iterator is exhausted.

    The *resp* is closed when the generator finishes or is cancelled.
    """
    seq = 0
    buf = b""
    try:
        async for chunk in resp.aiter_bytes():
            buf += chunk
            while b"\n\n" in buf:
                event_block, buf = buf.split(b"\n\n", 1)
                event, seq = _parse_sse_event(event_block, seq)
                if event is not None:
                    yield event
        if buf.strip():
            event, seq = _parse_sse_event(buf, seq)
            if event is not None:
                yield event
    finally:
        await resp.aclose()


async def _create_event_stream(
    client: httpx.AsyncClient, url: str, body: CreateResponseRequest,
) -> AsyncIterator[ResponseStreamEvent]:
    """Open a streaming request and yield validated event models.

    Combines ``stream_raw`` (HTTP transport) with
    ``_parse_response_stream`` (SSE parsing + validation) into a single
    async generator.
    """
    resp = await stream_raw(client, url, body)
    async for event in _parse_response_stream(resp):
        yield event


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------


class LlamacppClient(BaseProvider, ChatCompletions, Completions, Responses, Embeddings):
    name = "llamacpp"
    url_env = "LLAMACPP_URL"
    default_url = "http://llamacpp:8000"
    default_model_capabilities: ClassVar[list[str]] = ["chat", "completions", "embeddings"]
    models_path = "/v1/models"

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # -- ChatCompletions ------------------------------------------------------

    async def chat_completions(
        self, body: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = await forward(self._http_client, self._url("/v1/chat/completions"), body)
        return ChatCompletionResponse.model_validate(data)

    async def chat_completions_stream(
        self, body: ChatCompletionRequest,
    ) -> StreamingResponse:
        return await forward_stream(self._http_client, self._url("/v1/chat/completions"), body)

    async def chat_completions_stream_raw(
        self, body: ChatCompletionRequest,
    ) -> httpx.Response:
        """Return the raw streaming httpx response. Caller must close it."""
        return await stream_raw(self._http_client, self._url("/v1/chat/completions"), body)

    # -- Completions ----------------------------------------------------------

    async def completions(
        self, body: CompletionRequest,
    ) -> CompletionResponse:
        data = await forward(self._http_client, self._url("/v1/completions"), body)
        return CompletionResponse.model_validate(data)

    async def completions_stream(
        self, body: CompletionRequest,
    ) -> StreamingResponse:
        return await forward_stream(self._http_client, self._url("/v1/completions"), body)

    # -- Responses ------------------------------------------------------------

    async def create_response(
        self, body: CreateResponseRequest,
    ) -> CreateResponseResponse:
        data = await forward(self._http_client, self._url("/v1/responses"), body)
        return CreateResponseResponse.model_validate(data)

    def create_response_stream(
        self, body: CreateResponseRequest,
    ) -> AsyncIterator[ResponseStreamEvent]:
        return _create_event_stream(self._http_client, self._url("/v1/responses"), body)

    # -- Embeddings -----------------------------------------------------------

    async def embeddings(
        self, body: EmbeddingRequest,
    ) -> EmbeddingResponse:
        data = await forward(self._http_client, self._url("/v1/embeddings"), body)
        return EmbeddingResponse.model_validate(data)

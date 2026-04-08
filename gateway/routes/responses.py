"""POST /v1/responses — proxied to the responses provider.

Streaming responses are received as typed ``ResponseStreamEvent`` models
from the provider and serialized into SSE frames using FastAPI's
``EventSourceResponse`` and ``format_sse_event`` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import EventSourceResponse
from fastapi.sse import format_sse_event
from pydantic import TypeAdapter

from gateway.deps import ResponsesClient, backend_errors
from gateway.models import CreateResponseRequest, CreateResponseResponse
from gateway.models.responses import ResponseStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_stream_event_adapter = TypeAdapter(ResponseStreamEvent)

router = APIRouter()


async def _serialize_events(
    events: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[bytes]:
    """Serialize typed event models into SSE wire format.

    Each event is JSON-encoded via the ``ResponseStreamEvent``
    ``TypeAdapter`` and framed with ``format_sse_event``, which produces
    the ``event:`` and ``data:`` lines.  A ``data: [DONE]`` sentinel is
    appended after the iterator is exhausted, matching the OpenAI
    streaming contract.
    """
    async for event in events:
        data_str = _stream_event_adapter.dump_json(event).decode()
        yield format_sse_event(data_str=data_str, event=event.type)
    yield format_sse_event(data_str="[DONE]")


@router.post(
    "/v1/responses",
    response_model=CreateResponseResponse,
    # NB: do *not* set ``response_model_exclude_unset=True``.  The models in
    # ``gateway.models.responses`` deliberately declare every spec-required
    # field with a sensible default so the gateway can fill in fields that
    # the upstream provider (e.g. llama.cpp) omits.  ``exclude_unset`` would
    # strip exactly those defaults from the serialized response and break
    # strict OpenAI clients.  The streaming path already serializes via
    # ``TypeAdapter.dump_json`` (which does not apply ``exclude_unset``);
    # this keeps the non-streaming path consistent with that behavior.
)
async def create_response(
    body: CreateResponseRequest,
    client: ResponsesClient,
) -> CreateResponseResponse | EventSourceResponse:
    async with backend_errors("Response creation"):
        if body.stream:
            return EventSourceResponse(
                _serialize_events(client.create_response_stream(body)),
            )
        return await client.create_response(body)

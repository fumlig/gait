"""POST /v1/chat/completions — transparent proxy to the chat provider.

The gateway validates the request against ``ChatCompletionRequest``
(filling in defaults) and forwards the JSON payload unchanged to the
backend.  Multimodal content parts (``input_audio``, ``image_url``, …)
are passed through as-is — whether they are supported is up to the
backend model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from gateway.deps import ChatClient, backend_errors
from gateway.models import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_unset=True,
)
async def chat_completions(
    body: ChatCompletionRequest,
    client: ChatClient,
) -> ChatCompletionResponse | StreamingResponse:
    async with backend_errors("Chat completion"):
        if body.stream:
            return await client.chat_completions_stream(body)
        return await client.chat_completions(body)

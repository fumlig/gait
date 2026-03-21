from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from gateway_service.deps import CompletionsClient, backend_errors
from gateway_service.models import CompletionRequest, CompletionResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

router = APIRouter()


@router.post(
    "/v1/completions",
    response_model=CompletionResponse,
    response_model_exclude_unset=True,
)
async def completions(
    body: CompletionRequest,
    client: CompletionsClient,
) -> CompletionResponse | StreamingResponse:
    async with backend_errors("Text completion"):
        if body.stream:
            return await client.completions_stream(body)
        return await client.completions(body)

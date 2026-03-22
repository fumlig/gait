from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from gateway.deps import ResponsesClient, backend_errors
from gateway.models import CreateResponseRequest, CreateResponseResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

router = APIRouter()


@router.post(
    "/v1/responses",
    response_model=CreateResponseResponse,
    response_model_exclude_unset=True,
)
async def create_response(
    body: CreateResponseRequest,
    client: ResponsesClient,
) -> CreateResponseResponse | StreamingResponse:
    async with backend_errors("Response creation"):
        if body.stream:
            return await client.create_response_stream(body)
        return await client.create_response(body)

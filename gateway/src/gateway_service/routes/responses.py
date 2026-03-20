from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from gateway_service.models import CreateResponseRequest, CreateResponseResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

    from gateway_service.providers.protocols import Responses

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_responses_client(request: Request) -> Responses:
    client = getattr(request.app.state, "responses", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No responses backend configured.")
    return client


@router.post(
    "/v1/responses",
    response_model=CreateResponseResponse,
    response_model_exclude_unset=True,
)
async def create_response(
    request: Request,
    body: CreateResponseRequest,
) -> CreateResponseResponse | StreamingResponse:
    client = _get_responses_client(request)

    try:
        if body.stream:
            return await client.create_response_stream(body)
        return await client.create_response(body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Response creation failed")
        raise HTTPException(status_code=502, detail="Responses backend unavailable.") from exc

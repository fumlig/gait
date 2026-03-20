from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

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


@router.post("/v1/responses", response_model=None)
async def create_response(request: Request) -> JSONResponse | StreamingResponse:
    client = _get_responses_client(request)
    body = await request.json()

    try:
        if body.get("stream"):
            return await client.create_response_stream(body)
        result = await client.create_response(body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Response creation failed")
        raise HTTPException(status_code=502, detail="Responses backend unavailable.") from exc

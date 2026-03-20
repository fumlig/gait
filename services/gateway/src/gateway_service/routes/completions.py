from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

    from gateway_service.protocols import Completions

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_completions_client(request: Request) -> Completions:
    client = getattr(request.app.state, "completions", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No completions backend configured.")
    return client


@router.post("/v1/completions", response_model=None)
async def completions(request: Request) -> JSONResponse | StreamingResponse:
    client = _get_completions_client(request)
    body = await request.json()

    try:
        if body.get("stream"):
            return await client.completions_stream(body)
        result = await client.completions(body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Text completion failed")
        raise HTTPException(status_code=502, detail="Completions backend unavailable.") from exc

"""POST /v1/embeddings — proxied to llama.cpp server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from gateway_service.protocols import Embeddings

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_embeddings_client(request: Request) -> Embeddings:
    """Resolve the embeddings client from app state."""
    client = getattr(request.app.state, "embeddings", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No embeddings backend configured.")
    return client


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Create embeddings for the given input.

    Transparently proxies the request to the llama.cpp server backend.
    """
    client = _get_embeddings_client(request)
    body = await request.json()

    try:
        result = await client.embeddings(body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Embeddings request failed")
        raise HTTPException(status_code=502, detail="Embeddings backend unavailable.") from exc

"""POST /v1/embeddings — proxied to llama.cpp server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from gateway_service.clients.chat import ChatClient

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_chat_client(request: Request) -> ChatClient:
    """Resolve the chat client from app state."""
    client = getattr(request.app.state, "chat_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No chat backend configured.")
    return client


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Create embeddings for the given input.

    Transparently proxies the request to the llama.cpp server backend.
    """
    client = _get_chat_client(request)
    body = await request.json()

    try:
        result = await client.forward("/v1/embeddings", body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Embeddings request failed")
        raise HTTPException(status_code=502, detail="Chat backend unavailable.") from exc

"""POST /v1/completions — legacy completions, proxied to llama.cpp server."""

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


@router.post("/v1/completions")
async def completions(request: Request):
    """Create a text completion.

    Transparently proxies the request to the llama.cpp server backend.
    Supports both streaming (``"stream": true``) and non-streaming responses.
    """
    client = _get_chat_client(request)
    body = await request.json()

    try:
        if body.get("stream"):
            return await client.forward_stream("/v1/completions", body)
        result = await client.forward("/v1/completions", body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Text completion failed")
        raise HTTPException(status_code=502, detail="Chat backend unavailable.") from exc

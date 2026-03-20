"""POST /v1/responses — OpenAI Responses API, proxied to llama.cpp server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

    from gateway_service.clients.chat import ChatClient

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_chat_client(request: Request) -> ChatClient:
    """Resolve the chat client from app state."""
    client = getattr(request.app.state, "chat_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No chat backend configured.")
    return client


@router.post("/v1/responses", response_model=None)
async def create_response(request: Request) -> JSONResponse | StreamingResponse:
    """Create a model response.

    Transparently proxies the request to the llama.cpp server backend.
    Supports both streaming (``"stream": true``) and non-streaming responses.

    See https://platform.openai.com/docs/api-reference/responses
    """
    client = _get_chat_client(request)
    body = await request.json()

    try:
        if body.get("stream"):
            return await client.forward_stream("/v1/responses", body)
        result = await client.forward("/v1/responses", body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Response creation failed")
        raise HTTPException(status_code=502, detail="Chat backend unavailable.") from exc

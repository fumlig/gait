from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway_service.models import EmbeddingRequest, EmbeddingResponse

if TYPE_CHECKING:
    from gateway_service.providers.protocols import Embeddings

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_embeddings_client(request: Request) -> Embeddings:
    client = getattr(request.app.state, "embeddings", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No embeddings backend configured.")
    return client


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    response_model_exclude_unset=True,
)
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
) -> JSONResponse:
    client = _get_embeddings_client(request)
    payload = body.model_dump(exclude_unset=True)

    try:
        result = await client.embeddings(payload)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Embeddings request failed")
        raise HTTPException(status_code=502, detail="Embeddings backend unavailable.") from exc

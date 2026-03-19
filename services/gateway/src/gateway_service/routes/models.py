"""GET /v1/models -- merged model listing from all backends."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from gateway_service.models import ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """Return the cached model list from all backends."""
    cached_models = getattr(request.app.state, "models", [])
    return ModelListResponse(data=cached_models)

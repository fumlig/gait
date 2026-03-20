from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from gateway_service.models import ModelListResponse, ModelObject

logger = logging.getLogger(__name__)

router = APIRouter()

_MODELS_CACHE_TTL = 10  # seconds


async def _get_models(request: Request) -> list[ModelObject]:
    """Return the model list, refreshing if the cache is stale."""
    last_fetch = getattr(request.app.state, "models_fetched_at", 0)
    if time.monotonic() - last_fetch < _MODELS_CACHE_TTL:
        return getattr(request.app.state, "models", [])

    from gateway_service.main import fetch_all_models

    return await fetch_all_models(request.app)


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    models = await _get_models(request)
    return ModelListResponse(data=models)

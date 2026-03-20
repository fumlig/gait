"""GET /v1/models -- merged model listing from all backends."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from gateway_service.models import ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# How long (seconds) to cache the model list before re-fetching from backends.
_MODELS_CACHE_TTL = 10


async def _get_models(request: Request) -> list:
    """Return the model list, refreshing from backends if the cache is stale."""
    last_fetch = getattr(request.app.state, "models_fetched_at", 0)
    now = time.monotonic()

    if now - last_fetch < _MODELS_CACHE_TTL:
        return getattr(request.app.state, "models", [])

    # Refresh from backends
    from gateway_service.main import fetch_all_models

    return await fetch_all_models(request.app)


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """Return the model list from all backends (auto-refreshing)."""
    models = await _get_models(request)
    return ModelListResponse(data=models)

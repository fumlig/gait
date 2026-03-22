from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from gateway_service.models import ModelListResponse, ModelObject

if TYPE_CHECKING:
    from fastapi import FastAPI

    from gateway_service.providers.base import BaseProvider

logger = logging.getLogger(__name__)

router = APIRouter()

_refresh_lock = asyncio.Lock()
_CACHE_TTL = 10  # seconds


async def get_models(app: FastAPI) -> list[ModelObject]:
    """Return the cached model list, refreshing from providers if stale."""
    last_fetch: float = getattr(app.state, "models_fetched_at", 0)
    if time.monotonic() - last_fetch < _CACHE_TTL:
        return getattr(app.state, "models", [])

    async with _refresh_lock:
        # Re-check after acquiring lock — another coroutine may have refreshed.
        last_fetch = getattr(app.state, "models_fetched_at", 0)
        if time.monotonic() - last_fetch < _CACHE_TTL:
            return getattr(app.state, "models", [])

        providers: list[BaseProvider] = getattr(app.state, "providers", [])
        all_model_lists = await asyncio.gather(
            *(provider.fetch_models() for provider in providers)
        )

        seen: set[str] = set()
        merged: list[ModelObject] = []
        for model_list in all_model_lists:
            for model in model_list:
                if model.id not in seen:
                    seen.add(model.id)
                    merged.append(model)

        app.state.models = merged
        app.state.models_fetched_at = time.monotonic()
        return merged


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    result = await get_models(request.app)
    return ModelListResponse(data=result)

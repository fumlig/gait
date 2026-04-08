from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from gateway.deps import backend_errors
from gateway.models import (
    LoadModelRequest,
    LoadModelResponse,
    ModelListResponse,
    ModelObject,
    UnloadModelRequest,
    UnloadModelResponse,
)
from gateway.providers.protocols import ModelManagement

if TYPE_CHECKING:
    from fastapi import FastAPI

    from gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)

router = APIRouter()

_refresh_lock = asyncio.Lock()
_CACHE_TTL = 10  # seconds


async def _refresh_loaded_status(
    models: list[ModelObject],
    providers: list[BaseProvider],
) -> None:
    """Update ``loaded`` on each model based on provider health checks."""
    health_results = await asyncio.gather(
        *(provider.check_health() for provider in providers),
    )
    healthy: dict[str, bool] = {
        provider.name: result.get("status") == "healthy"
        for provider, result in zip(providers, health_results)
    }
    for model in models:
        if model.owned_by in healthy:
            model.loaded = healthy[model.owned_by]


async def get_models(app: FastAPI) -> list[ModelObject]:
    """Return the cached model list, refreshing from providers if stale.

    The model catalog (IDs, capabilities) is cached with a TTL.  When a
    provider fails to respond, its previously cached models are preserved
    (rather than disappearing) so the catalog remains stable.

    The ``loaded`` status is always refreshed via provider health checks
    so that callers never see a stale readiness state.
    """
    last_fetch: float = getattr(app.state, "models_fetched_at", 0)

    if time.monotonic() - last_fetch >= _CACHE_TTL:
        async with _refresh_lock:
            # Re-check after acquiring lock — another coroutine may have refreshed.
            last_fetch = getattr(app.state, "models_fetched_at", 0)
            if time.monotonic() - last_fetch >= _CACHE_TTL:
                providers: list[BaseProvider] = getattr(app.state, "providers", [])
                all_model_lists = await asyncio.gather(
                    *(provider.fetch_models() for provider in providers)
                )

                # Build a map of provider → fresh models.  When a
                # provider returns nothing (error / 503), fall back to
                # whatever was previously cached for that provider so
                # models don't vanish during transient unavailability.
                prev_models: list[ModelObject] = getattr(app.state, "models", [])
                prev_by_provider: dict[str, list[ModelObject]] = {}
                for m in prev_models:
                    prev_by_provider.setdefault(m.owned_by, []).append(m)

                seen: set[str] = set()
                merged: list[ModelObject] = []
                for provider, model_list in zip(providers, all_model_lists):
                    effective = model_list or prev_by_provider.get(provider.name, [])
                    for model in effective:
                        if model.id not in seen:
                            seen.add(model.id)
                            merged.append(model)

                app.state.models = merged
                app.state.models_fetched_at = time.monotonic()

    models: list[ModelObject] = getattr(app.state, "models", [])
    providers = getattr(app.state, "providers", [])
    await _refresh_loaded_status(models, providers)
    return models


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    result = await get_models(request.app)
    return ModelListResponse(data=result)


# ---------------------------------------------------------------------------
# Model management (gait extension — not in the OpenAI API)
# ---------------------------------------------------------------------------


def _invalidate_models_cache(app: FastAPI) -> None:
    """Force the next ``get_models`` call to refresh from providers."""
    app.state.models_fetched_at = 0.0


async def _find_manager_for_model(
    app: FastAPI, model_id: str,
) -> ModelManagement:
    """Look up the provider that owns ``model_id`` and return it as a manager.

    Raises ``HTTPException(404)`` if no provider owns the model, and
    ``HTTPException(400)`` if the owning provider does not implement
    the ``ModelManagement`` protocol.
    """
    models: list[ModelObject] = await get_models(app)
    owner: str | None = None
    for m in models:
        if m.id == model_id:
            owner = m.owned_by
            break
    if owner is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_id}'.",
        )

    providers: list[BaseProvider] = getattr(app.state, "providers", [])
    for provider in providers:
        if provider.name != owner:
            continue
        if not isinstance(provider, ModelManagement):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider '{owner}' does not support model "
                    f"management (load/unload)."
                ),
            )
        return provider

    raise HTTPException(
        status_code=404,
        detail=f"No provider '{owner}' registered for model '{model_id}'.",
    )


@router.post(
    "/v1/models/load",
    response_model=LoadModelResponse,
    response_model_exclude_unset=True,
)
async def load_model(
    body: LoadModelRequest,
    request: Request,
) -> LoadModelResponse:
    manager = await _find_manager_for_model(request.app, body.model)
    async with backend_errors("Model load"):
        result = await manager.load_model(body.model)
    _invalidate_models_cache(request.app)
    return result


@router.post(
    "/v1/models/unload",
    response_model=UnloadModelResponse,
    response_model_exclude_unset=True,
)
async def unload_model(
    body: UnloadModelRequest,
    request: Request,
) -> UnloadModelResponse:
    manager = await _find_manager_for_model(request.app, body.model)
    async with backend_errors("Model unload"):
        result = await manager.unload_model(body.model)
    _invalidate_models_cache(request.app)
    return result

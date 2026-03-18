"""GET /v1/models -- merged model listing from all backends."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from gateway_service.config import settings
from gateway_service.models import ModelListResponse, ModelObject
from gateway_service.proxy import get_client

logger = logging.getLogger(__name__)

router = APIRouter()


async def _fetch_models(backend_url: str, backend_name: str) -> list[ModelObject]:
    """Fetch /v1/models from a single backend, returning [] on failure."""
    client = get_client()
    url = f"{backend_url.rstrip('/')}/v1/models"
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Non-200 from %s (%s): HTTP %d", backend_name, url, resp.status_code)
            return []
        data = resp.json()
        return [ModelObject(**m) for m in data.get("data", [])]
    except Exception:
        logger.warning("Failed to fetch models from %s (%s)", backend_name, url, exc_info=True)
        return []


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """Merge /v1/models responses from all backend services."""
    # Deduplicate backends by URL
    backends = {
        settings.chatterbox_url: "chatterbox",
        settings.whisperx_url: "whisperx",
    }

    tasks = [_fetch_models(url, name) for url, name in backends.items()]
    results = await asyncio.gather(*tasks)

    # Merge and deduplicate by model ID
    seen: set[str] = set()
    merged: list[ModelObject] = []
    for model_list in results:
        for model in model_list:
            if model.id not in seen:
                seen.add(model.id)
                merged.append(model)

    return ModelListResponse(data=merged)

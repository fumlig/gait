"""GET /health — service health check."""

from __future__ import annotations

from fastapi import APIRouter

from chatterbox_service.engine import engine
from chatterbox_service.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_loaded,
        models=engine.loaded_models,
    )

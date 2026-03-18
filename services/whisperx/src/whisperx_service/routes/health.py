"""GET /health -- service health check."""

from __future__ import annotations

from fastapi import APIRouter

from whisperx_service.engine import engine
from whisperx_service.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model_loaded=engine.is_loaded)

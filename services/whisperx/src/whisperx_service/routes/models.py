"""GET /v1/models -- OpenAI-compatible models listing."""

from __future__ import annotations

from fastapi import APIRouter

from whisperx_service.engine import engine
from whisperx_service.models import ModelListResponse, ModelObject

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    models = [ModelObject(id=model_id) for model_id in engine.list_available_models()]
    return ModelListResponse(data=models)

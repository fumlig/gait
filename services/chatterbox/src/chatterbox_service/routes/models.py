"""GET /v1/models — OpenAI-compatible models listing."""

from __future__ import annotations

from fastapi import APIRouter

from chatterbox_service.models import ModelListResponse, ModelObject

router = APIRouter()

AVAILABLE_MODELS = [
    ModelObject(id="chatterbox-turbo", owned_by="resemble-ai"),
    ModelObject(id="chatterbox", owned_by="resemble-ai"),
    ModelObject(id="chatterbox-multilingual", owned_by="resemble-ai"),
]


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=AVAILABLE_MODELS)

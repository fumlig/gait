from __future__ import annotations

from fastapi import APIRouter

from gateway_service.deps import EmbeddingsClient, backend_errors
from gateway_service.models import EmbeddingRequest, EmbeddingResponse

router = APIRouter()


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    response_model_exclude_unset=True,
)
async def embeddings(
    body: EmbeddingRequest,
    client: EmbeddingsClient,
) -> EmbeddingResponse:
    async with backend_errors("Embeddings request"):
        return await client.embeddings(body)

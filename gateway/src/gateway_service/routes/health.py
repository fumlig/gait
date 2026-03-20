from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from gateway_service.models import GatewayHealthResponse

if TYPE_CHECKING:
    from gateway_service.providers.base import BaseProvider

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=GatewayHealthResponse)
async def health(request: Request) -> GatewayHealthResponse:
    providers: list[BaseProvider] = getattr(request.app.state, "providers", [])

    provider_status: dict[str, str] = {}
    for provider in providers:
        try:
            result = await provider.check_health()
            provider_status[provider.name] = result.get("status", "unknown")
        except Exception:
            provider_status[provider.name] = "unreachable"

    if not provider_status or all(v == "healthy" for v in provider_status.values()):
        overall = "ok"
    else:
        overall = "degraded"

    return GatewayHealthResponse(status=overall, backends=provider_status)

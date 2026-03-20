from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from gateway_service.models import GatewayHealthResponse

if TYPE_CHECKING:
    from gateway_service.backends.base import BaseBackend

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=GatewayHealthResponse)
async def health(request: Request) -> GatewayHealthResponse:
    backends: list[BaseBackend] = getattr(request.app.state, "backends", [])

    backend_status: dict[str, str] = {}
    for backend in backends:
        try:
            result = await backend.check_health()
            backend_status[backend.name] = result.get("status", "unknown")
        except Exception:
            backend_status[backend.name] = "unreachable"

    if not backend_status or all(v == "healthy" for v in backend_status.values()):
        overall = "ok"
    else:
        overall = "degraded"

    return GatewayHealthResponse(status=overall, backends=backend_status)

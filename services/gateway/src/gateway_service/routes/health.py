"""GET /health -- gateway and backend health checks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from gateway_service.models import GatewayHealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=GatewayHealthResponse)
async def health(request: Request) -> GatewayHealthResponse:
    """Report gateway health and status of each backend."""
    speech_client = getattr(request.app.state, "speech_client", None)
    transcription_client = getattr(request.app.state, "transcription_client", None)
    chat_client = getattr(request.app.state, "chat_client", None)

    backend_status: dict[str, str] = {}

    # Check each remote backend's health (gracefully handle missing clients)
    for name, client in [
        ("speech", speech_client),
        ("transcription", transcription_client),
        ("chat", chat_client),
    ]:
        if client is None:
            backend_status[name] = "not_configured"
            continue
        try:
            result = await client.health()
            backend_status[name] = result.get("status", "unknown")
        except Exception:
            backend_status[name] = "unreachable"

    overall = "ok" if all(v == "healthy" for v in backend_status.values()) else "degraded"

    return GatewayHealthResponse(status=overall, backends=backend_status)

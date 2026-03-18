"""GET /health -- gateway and backend health checks."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from gateway_service.config import settings
from gateway_service.models import HealthResponse
from gateway_service.proxy import get_client

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_backend(backend_url: str, backend_name: str) -> tuple[str, str]:
    """Check health of a single backend. Returns (name, status)."""
    client = get_client()
    url = f"{backend_url.rstrip('/')}/health"
    try:
        resp = await client.get(url, timeout=5.0)
        if resp.status_code == 200:
            return backend_name, "healthy"
        return backend_name, f"unhealthy (HTTP {resp.status_code})"
    except Exception as exc:
        return backend_name, f"unreachable ({type(exc).__name__})"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report gateway health and status of each backend."""
    backends = {
        settings.chatterbox_url: "chatterbox",
        settings.whisperx_url: "whisperx",
    }

    tasks = [_check_backend(url, name) for url, name in backends.items()]
    results = await asyncio.gather(*tasks)

    backend_status = dict(results)
    overall = "ok" if all(v == "healthy" for v in backend_status.values()) else "degraded"

    return HealthResponse(status=overall, backends=backend_status)

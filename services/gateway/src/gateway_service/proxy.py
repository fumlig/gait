"""Reverse proxy utilities for forwarding requests to backend services."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from gateway_service.config import settings

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Shared async client — created during app lifespan, closed on shutdown.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Return the shared httpx async client."""
    if _client is None:
        raise RuntimeError("HTTP client not initialised — app lifespan not started")
    return _client


def create_client() -> httpx.AsyncClient:
    """Create and store the shared httpx client."""
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.proxy_timeout, connect=10.0),
        follow_redirects=False,
    )
    return _client


async def close_client() -> None:
    """Close the shared httpx client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def proxy_request(backend_url: str, request: Request) -> httpx.Response:
    """Forward an incoming request to a backend service.

    Preserves method, path, query string, headers, and body.
    """
    client = get_client()

    # Build the target URL: backend base + original path + query string
    target_url = f"{backend_url.rstrip('/')}{request.url.path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # Forward headers (strip hop-by-hop and host)
    headers = dict(request.headers)
    for skip in ("host", "transfer-encoding", "connection"):
        headers.pop(skip, None)

    body = await request.body()

    logger.info("Proxying %s %s -> %s", request.method, request.url.path, target_url)

    resp = await client.request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
    )
    return resp


def httpx_to_fastapi_response(resp: httpx.Response) -> Response:
    """Convert an httpx response to a FastAPI Response."""
    from fastapi.responses import Response

    # Forward response headers, excluding hop-by-hop headers
    headers = {}
    for key, value in resp.headers.items():
        lower = key.lower()
        if lower not in ("transfer-encoding", "connection", "content-encoding", "content-length"):
            headers[key] = value

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type"),
    )

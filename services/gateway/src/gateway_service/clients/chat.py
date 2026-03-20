"""Chat client — transparent proxy to the llama.cpp server backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from gateway_service.models import ModelObject

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class ChatClient:
    """Typed HTTP client for the llama.cpp server backend.

    Since llama-server is already OpenAI-compatible, this client acts as a
    transparent proxy — forwarding requests and streaming responses without
    any schema transformation.

    Proxied endpoints:
    - POST /v1/chat/completions  (JSON body → JSON or SSE stream)
    - POST /v1/completions       (JSON body → JSON or SSE stream)
    - POST /v1/embeddings        (JSON body → JSON)
    - GET  /v1/models            (model list)
    - GET  /health               (health check)
    """

    def __init__(self, *, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def forward(self, path: str, body: dict) -> dict:
        """Forward a JSON request and return the JSON response.

        Used for non-streaming completions and embeddings.
        """
        url = f"{self._base_url}{path}"
        resp = await self._client.post(url, json=body)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp.json()

    async def forward_stream(self, path: str, body: dict) -> StreamingResponse:
        """Forward a JSON request and return a streaming response.

        Opens an httpx stream to the backend and wraps it in a Starlette
        ``StreamingResponse``.  The stream is kept open until the client
        finishes consuming the response, then cleaned up in the generator's
        ``finally`` block.
        """
        url = f"{self._base_url}{path}"

        req = self._client.build_request("POST", url, json=body)
        resp = await self._client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type", "text/event-stream")

        async def _generate():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(_generate(), media_type=content_type)

    async def list_models(self) -> list[ModelObject]:
        """Fetch GET /v1/models from the backend."""
        url = f"{self._base_url}/v1/models"
        try:
            resp = await self._client.get(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(
                    "Non-200 from chat backend (%s): HTTP %d", url, resp.status_code
                )
                return []
            data = resp.json()
            return [ModelObject(**m) for m in data.get("data", [])]
        except Exception:
            logger.warning("Failed to fetch models from chat backend (%s)", url, exc_info=True)
            return []

    async def stream_raw(self, path: str, body: dict) -> httpx.Response:
        """Forward a JSON request and return the raw streaming httpx response.

        Unlike ``forward_stream``, the caller receives the raw ``httpx.Response``
        for custom processing (e.g. interleaving audio).  The caller **must**
        close the response when done (typically via ``await resp.aclose()``
        in a ``finally`` block).
        """
        url = f"{self._base_url}{path}"
        req = self._client.build_request("POST", url, json=body)
        resp = await self._client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp

    async def health(self) -> dict:
        """Check GET /health on the backend."""
        url = f"{self._base_url}/health"
        try:
            resp = await self._client.get(url, timeout=5.0)
            if resp.status_code == 200:
                return {"status": "healthy"}
            return {"status": "unhealthy", "detail": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"status": "unreachable", "detail": f"{type(exc).__name__}: {exc}"}

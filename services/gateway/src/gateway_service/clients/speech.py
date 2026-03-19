"""Speech client — calls the chatterbox backend's /synthesize endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException

from gateway_service.models import ModelObject

if TYPE_CHECKING:
    import httpx

    from gateway_service.models import SpeechRequest

logger = logging.getLogger(__name__)


class SpeechClient:
    """Typed HTTP client for the speech (TTS) backend.

    Calls chatterbox's RPC endpoints:
    - POST /synthesize  (JSON body → audio/wav binary)
    - GET /models       (model list)
    - GET /health       (health check)
    """

    def __init__(self, *, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Send a synthesize request and return (wav_bytes, content_type).

        Maps the OpenAI-compatible ``input`` field to the backend's ``text``
        field.  Strips ``response_format`` since the backend always returns
        WAV — format conversion is the gateway's responsibility.
        """
        url = f"{self._base_url}/synthesize"

        # Build payload: exclude None values so backend applies its defaults.
        payload = request.model_dump(exclude_none=True)

        # Map OpenAI field name → backend field name
        payload["text"] = payload.pop("input")

        # Strip response_format — backend always returns WAV
        payload.pop("response_format", None)

        resp = await self._client.post(url, json=payload)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type", "audio/wav")
        return resp.content, content_type

    async def list_models(self) -> list[ModelObject]:
        """Fetch GET /models from the backend."""
        url = f"{self._base_url}/models"
        try:
            resp = await self._client.get(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("Non-200 from speech backend (%s): HTTP %d", url, resp.status_code)
                return []
            data = resp.json()
            return [ModelObject(**m) for m in data.get("data", [])]
        except Exception:
            logger.warning("Failed to fetch models from speech backend (%s)", url, exc_info=True)
            return []

    async def health(self) -> dict:
        """Check GET /health on the backend."""
        url = f"{self._base_url}/health"
        resp = await self._client.get(url, timeout=5.0)
        if resp.status_code == 200:
            return {"status": "healthy"}
        return {"status": "unhealthy", "detail": f"HTTP {resp.status_code}"}

"""Voice client — calls the voice service's /voices endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException

from gateway_service.models import Voice

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class VoiceClient:
    """Typed HTTP client for the voice management service.

    Calls the voice service's endpoints:
    - GET    /voices         (list all voices)
    - GET    /voices/{name}  (get a single voice)
    - POST   /voices         (create a voice, multipart)
    - DELETE /voices/{name}  (delete a voice)
    - GET    /health         (health check)
    """

    def __init__(self, *, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def list_voices(self) -> list[Voice]:
        """GET /voices on the backend."""
        url = f"{self._base_url}/voices"
        resp = await self._client.get(url)
        self._raise_for_status(resp)
        data = resp.json()
        return [Voice(**v) for v in data.get("data", [])]

    async def get_voice(self, name: str) -> Voice:
        """GET /voices/{name} on the backend."""
        url = f"{self._base_url}/voices/{name}"
        resp = await self._client.get(url)
        self._raise_for_status(resp)
        return Voice(**resp.json())

    async def create_voice(self, name: str, audio_data: bytes) -> Voice:
        """POST /voices on the backend (multipart)."""
        url = f"{self._base_url}/voices"
        resp = await self._client.post(
            url,
            data={"name": name},
            files={"file": ("voice.wav", audio_data, "audio/wav")},
        )
        self._raise_for_status(resp)
        return Voice(**resp.json())

    async def delete_voice(self, name: str) -> dict:
        """DELETE /voices/{name} on the backend."""
        url = f"{self._base_url}/voices/{name}"
        resp = await self._client.delete(url)
        self._raise_for_status(resp)
        return resp.json()

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

    @staticmethod
    def _raise_for_status(resp) -> None:
        """Raise HTTPException if the backend returned a non-2xx status."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

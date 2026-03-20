"""Chatterbox provider client — TTS via the /synthesize endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from fastapi import HTTPException

from gateway_service.providers.base import BaseProvider
from gateway_service.providers.protocols import AudioSpeech

if TYPE_CHECKING:
    from gateway_service.models import SpeechRequest

logger = logging.getLogger(__name__)


class ChatterboxClient(BaseProvider, AudioSpeech):
    name = "chatterbox"
    env_var = "CHATTERBOX_URL"
    default_model_capabilities: ClassVar[list[str]] = ["speech"]

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Send a synthesize request and return (wav_bytes, content_type)."""
        url = f"{self._base_url}/synthesize"

        # Build payload, mapping OpenAI's "input" to backend's "text"
        payload = request.model_dump(exclude_none=True)
        payload["text"] = payload.pop("input")
        payload.pop("response_format", None)  # backend always returns WAV

        resp = await self._http_client.post(url, json=payload)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type", "audio/wav")
        return resp.content, content_type

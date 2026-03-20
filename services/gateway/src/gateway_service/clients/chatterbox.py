"""Chatterbox backend client — calls the chatterbox TTS /synthesize endpoint.

Implements :class:`~gateway_service.clients.protocols.AudioSpeech`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from fastapi import HTTPException

from gateway_service.clients.base import BaseBackend

if TYPE_CHECKING:
    from gateway_service.models import SpeechRequest

logger = logging.getLogger(__name__)


class ChatterboxClient(BaseBackend):
    """Typed HTTP client for the chatterbox speech (TTS) backend.

    Calls chatterbox's RPC endpoints:
    - POST /synthesize  (JSON body → audio/wav binary)
    """

    name = "chatterbox"
    env_var = "CHATTERBOX_URL"
    default_model_capabilities: ClassVar[list[str]] = ["speech"]

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

        resp = await self._http_client.post(url, json=payload)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type", "audio/wav")
        return resp.content, content_type

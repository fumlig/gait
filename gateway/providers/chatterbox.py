"""Chatterbox provider client — TTS via the /synthesize endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from gateway.models import LoadModelResponse, UnloadModelResponse
from gateway.providers.base import BaseProvider, status_from_payload
from gateway.providers.protocols import AudioSpeech, ModelManagement

if TYPE_CHECKING:
    from gateway.models import SpeechRequest

logger = logging.getLogger(__name__)


class ChatterboxClient(BaseProvider, AudioSpeech, ModelManagement):
    name = "chatterbox"
    url_env = "CHATTERBOX_URL"
    default_url = "http://chatterbox:8000"
    default_model_capabilities: ClassVar[list[str]] = ["speech"]

    # -- ModelManagement ------------------------------------------------------

    async def load_model(self, model: str) -> LoadModelResponse:
        payload = await self._post_model_action("/models/load", model)
        return LoadModelResponse(
            success=bool(payload.get("success", True)),
            model=str(payload.get("model", model)),
            status=status_from_payload(payload.get("status")),
        )

    async def unload_model(self, model: str) -> UnloadModelResponse:
        payload = await self._post_model_action("/models/unload", model)
        return UnloadModelResponse(
            success=bool(payload.get("success", True)),
            model=str(payload.get("model", model)),
            status=status_from_payload(payload.get("status")),
        )

    async def _post_model_action(self, path: str, model: str) -> dict:
        resp = await self._http_client.post(
            f"{self._base_url}{path}", json={"model": model},
        )
        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json() if resp.content else {}

    def _build_payload(self, request: SpeechRequest) -> dict:
        """Build the backend JSON payload from an OpenAI SpeechRequest."""
        payload = request.model_dump(exclude_none=True)
        payload["text"] = payload.pop("input")
        payload.pop("response_format", None)  # backend always returns WAV
        return payload

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Send a synthesize request and return (wav_bytes, content_type)."""
        url = f"{self._base_url}/synthesize"
        payload = self._build_payload(request)

        resp = await self._http_client.post(url, json=payload)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_type = resp.headers.get("content-type", "audio/wav")
        return resp.content, content_type

    async def synthesize_stream(self, request: SpeechRequest) -> StreamingResponse:
        """Proxy WAV bytes from the backend as a streaming response.

        The chatterbox model generates the full waveform atomically,
        but streaming the HTTP response avoids double-buffering in
        the gateway.
        """
        url = f"{self._base_url}/synthesize"
        payload = self._build_payload(request)

        req = self._http_client.build_request("POST", url, json=payload)
        resp = await self._http_client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        content_length = resp.headers.get("content-length")
        headers: dict[str, str] = {
            "Content-Disposition": 'inline; filename="speech.wav"',
        }
        if content_length:
            headers["Content-Length"] = content_length

        async def _proxy():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(
            _proxy(),
            media_type="audio/wav",
            headers=headers,
        )

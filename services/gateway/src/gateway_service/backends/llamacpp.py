"""Llama.cpp backend client — transparent proxy to the llama.cpp server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from gateway_service.backends.base import BaseBackend
from gateway_service.protocols import ChatCompletions, Completions, Embeddings, Responses

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class LlamacppClient(BaseBackend, ChatCompletions, Completions, Responses, Embeddings):
    name = "llamacpp"
    env_var = "LLAMACPP_URL"
    default_model_capabilities: ClassVar[list[str]] = ["chat", "completions", "embeddings"]
    models_path = "/v1/models"

    async def chat_completions(self, body: dict) -> dict:
        return await self._forward("/v1/chat/completions", body)

    async def chat_completions_stream(self, body: dict) -> StreamingResponse:
        return await self._forward_stream("/v1/chat/completions", body)

    async def chat_completions_stream_raw(self, body: dict) -> httpx.Response:
        """Return the raw streaming httpx response. Caller must close it."""
        return await self._stream_raw("/v1/chat/completions", body)

    async def completions(self, body: dict) -> dict:
        return await self._forward("/v1/completions", body)

    async def completions_stream(self, body: dict) -> StreamingResponse:
        return await self._forward_stream("/v1/completions", body)

    async def create_response(self, body: dict) -> dict:
        return await self._forward("/v1/responses", body)

    async def create_response_stream(self, body: dict) -> StreamingResponse:
        return await self._forward_stream("/v1/responses", body)

    async def embeddings(self, body: dict) -> dict:
        return await self._forward("/v1/embeddings", body)

    async def _forward(self, path: str, body: dict) -> dict:
        url = f"{self._base_url}{path}"
        resp = await self._http_client.post(url, json=body)
        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json()

    async def _forward_stream(self, path: str, body: dict) -> StreamingResponse:
        url = f"{self._base_url}{path}"
        req = self._http_client.build_request("POST", url, json=body)
        resp = await self._http_client.send(req, stream=True)

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

    async def _stream_raw(self, path: str, body: dict) -> httpx.Response:
        url = f"{self._base_url}{path}"
        req = self._http_client.build_request("POST", url, json=body)
        resp = await self._http_client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp

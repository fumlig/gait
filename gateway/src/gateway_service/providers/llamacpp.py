"""Llama.cpp provider client — typed proxy to the llama.cpp server.

Accepts typed request models, serialises them to JSON for the HTTP call,
and parses the response JSON back into typed response models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from gateway_service.models import (
    ChatCompletionResponse,
    CompletionResponse,
    CreateResponseResponse,
    EmbeddingResponse,
)
from gateway_service.providers.base import BaseProvider
from gateway_service.providers.protocols import (
    ChatCompletions,
    Completions,
    Embeddings,
    Responses,
)

if TYPE_CHECKING:
    import httpx
    from pydantic import BaseModel

    from gateway_service.models import (
        ChatCompletionRequest,
        CompletionRequest,
        CreateResponseRequest,
        EmbeddingRequest,
    )

logger = logging.getLogger(__name__)


class LlamacppClient(BaseProvider, ChatCompletions, Completions, Responses, Embeddings):
    name = "llamacpp"
    env_var = "LLAMACPP_URL"
    default_model_capabilities: ClassVar[list[str]] = ["chat", "completions", "embeddings"]
    models_path = "/v1/models"

    # -- ChatCompletions ------------------------------------------------------

    async def chat_completions(
        self, body: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = await self._forward("/v1/chat/completions", body)
        return ChatCompletionResponse.model_validate(data)

    async def chat_completions_stream(
        self, body: ChatCompletionRequest,
    ) -> StreamingResponse:
        return await self._forward_stream("/v1/chat/completions", body)

    async def chat_completions_stream_raw(
        self, body: ChatCompletionRequest,
    ) -> httpx.Response:
        """Return the raw streaming httpx response. Caller must close it."""
        return await self._stream_raw("/v1/chat/completions", body)

    # -- Completions ----------------------------------------------------------

    async def completions(
        self, body: CompletionRequest,
    ) -> CompletionResponse:
        data = await self._forward("/v1/completions", body)
        return CompletionResponse.model_validate(data)

    async def completions_stream(
        self, body: CompletionRequest,
    ) -> StreamingResponse:
        return await self._forward_stream("/v1/completions", body)

    # -- Responses ------------------------------------------------------------

    async def create_response(
        self, body: CreateResponseRequest,
    ) -> CreateResponseResponse:
        data = await self._forward("/v1/responses", body)
        return CreateResponseResponse.model_validate(data)

    async def create_response_stream(
        self, body: CreateResponseRequest,
    ) -> StreamingResponse:
        return await self._forward_stream("/v1/responses", body)

    # -- Embeddings -----------------------------------------------------------

    async def embeddings(
        self, body: EmbeddingRequest,
    ) -> EmbeddingResponse:
        data = await self._forward("/v1/embeddings", body)
        return EmbeddingResponse.model_validate(data)

    # -- transport helpers ----------------------------------------------------

    async def _forward(
        self, path: str, body: BaseModel,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        payload = body.model_dump(exclude_unset=True)
        resp = await self._http_client.post(url, json=payload)
        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json()

    async def _forward_stream(
        self, path: str, body: BaseModel,
    ) -> StreamingResponse:
        url = f"{self._base_url}{path}"
        payload = body.model_dump(exclude_unset=True)
        req = self._http_client.build_request("POST", url, json=payload)
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

    async def _stream_raw(
        self, path: str, body: BaseModel,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        payload = body.model_dump(exclude_unset=True)
        req = self._http_client.build_request("POST", url, json=payload)
        resp = await self._http_client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp

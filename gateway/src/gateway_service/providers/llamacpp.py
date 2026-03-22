"""Llama.cpp provider client — typed proxy to the llama.cpp server.

Each method maps an OpenAI endpoint to the corresponding backend path
and response model.  The actual HTTP transport is handled by the
``forward``, ``forward_stream``, and ``stream_raw`` helpers from
``providers.transport``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

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
from gateway_service.providers.transport import forward, forward_stream, stream_raw

if TYPE_CHECKING:
    import httpx
    from starlette.responses import StreamingResponse

    from gateway_service.models import (
        ChatCompletionRequest,
        CompletionRequest,
        CreateResponseRequest,
        EmbeddingRequest,
    )


class LlamacppClient(BaseProvider, ChatCompletions, Completions, Responses, Embeddings):
    name = "llamacpp"
    env_var = "LLAMACPP_URL"
    default_model_capabilities: ClassVar[list[str]] = ["chat", "completions", "embeddings"]
    models_path = "/v1/models"

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # -- ChatCompletions ------------------------------------------------------

    async def chat_completions(
        self, body: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = await forward(self._http_client, self._url("/v1/chat/completions"), body)
        return ChatCompletionResponse.model_validate(data)

    async def chat_completions_stream(
        self, body: ChatCompletionRequest,
    ) -> StreamingResponse:
        return await forward_stream(self._http_client, self._url("/v1/chat/completions"), body)

    async def chat_completions_stream_raw(
        self, body: ChatCompletionRequest,
    ) -> httpx.Response:
        """Return the raw streaming httpx response. Caller must close it."""
        return await stream_raw(self._http_client, self._url("/v1/chat/completions"), body)

    # -- Completions ----------------------------------------------------------

    async def completions(
        self, body: CompletionRequest,
    ) -> CompletionResponse:
        data = await forward(self._http_client, self._url("/v1/completions"), body)
        return CompletionResponse.model_validate(data)

    async def completions_stream(
        self, body: CompletionRequest,
    ) -> StreamingResponse:
        return await forward_stream(self._http_client, self._url("/v1/completions"), body)

    # -- Responses ------------------------------------------------------------

    async def create_response(
        self, body: CreateResponseRequest,
    ) -> CreateResponseResponse:
        data = await forward(self._http_client, self._url("/v1/responses"), body)
        return CreateResponseResponse.model_validate(data)

    async def create_response_stream(
        self, body: CreateResponseRequest,
    ) -> StreamingResponse:
        return await forward_stream(self._http_client, self._url("/v1/responses"), body)

    # -- Embeddings -----------------------------------------------------------

    async def embeddings(
        self, body: EmbeddingRequest,
    ) -> EmbeddingResponse:
        data = await forward(self._http_client, self._url("/v1/embeddings"), body)
        return EmbeddingResponse.model_validate(data)

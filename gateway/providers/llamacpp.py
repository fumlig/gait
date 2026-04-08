"""Llama.cpp provider client — typed proxy to the llama.cpp server.

Each method maps an OpenAI endpoint to the corresponding backend path
and response model.  The actual HTTP transport is handled by the
``forward``, ``forward_stream``, and ``stream_raw`` helpers from
``providers.transport``.

For the Responses API stream, each SSE event is parsed from the
upstream byte stream, validated through the ``ResponseStreamEvent``
discriminated union (filling spec-required defaults), and yielded as
a typed Pydantic model.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar, cast

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from gateway.models import (
    ChatCompletionResponse,
    CompletionResponse,
    CreateResponseResponse,
    EmbeddingResponse,
    LoadModelResponse,
    ModelObject,
    ModelStatus,
    UnloadModelResponse,
)
from gateway.models.responses import ResponseStreamEvent
from gateway.providers.base import BaseProvider
from gateway.providers.protocols import (
    ChatCompletions,
    Completions,
    Embeddings,
    ModelManagement,
    Responses,
)
from gateway.providers.transport import forward, forward_stream, stream_raw

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from starlette.responses import StreamingResponse

    from gateway.models import (
        ChatCompletionRequest,
        CompletionRequest,
        CreateResponseRequest,
        EmbeddingRequest,
    )
    from gateway.models.common import ModelStatusValue

logger = logging.getLogger(__name__)

_stream_event_adapter = TypeAdapter(ResponseStreamEvent)


# ---------------------------------------------------------------------------
# Router mode status parsing
# ---------------------------------------------------------------------------


def _parse_llamacpp_status(raw: object) -> ModelStatus | None:
    """Convert llama-server's ``status`` object into a ``ModelStatus``.

    Accepts ``None`` or a dict; unknown ``value`` strings fall back to
    ``"unloaded"``.  Returns ``None`` if ``raw`` is not a dict — the
    caller decides whether to omit the field or substitute a default.
    """
    if not isinstance(raw, dict):
        return None
    d = cast("dict[str, object]", raw)

    raw_value = d.get("value")
    value: ModelStatusValue = (
        cast("ModelStatusValue", raw_value)
        if raw_value in ("unloaded", "loading", "loaded", "sleeping")
        else "unloaded"
    )

    args_raw = d.get("args")
    args = args_raw if isinstance(args_raw, list) else []

    exit_code_raw = d.get("exit_code")
    exit_code: int | None
    if isinstance(exit_code_raw, int):
        exit_code = exit_code_raw
    elif exit_code_raw is None:
        exit_code = None
    else:
        try:
            exit_code = int(cast("str | int | float", exit_code_raw))
        except (TypeError, ValueError):
            exit_code = None

    return ModelStatus(
        value=value,
        args=[str(a) for a in args],
        failed=bool(d.get("failed") or False),
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Responses-stream SSE → typed event parsing
# ---------------------------------------------------------------------------


def _parse_sse_event(
    block: bytes,
    seq: int,
) -> tuple[ResponseStreamEvent | None, int]:
    """Parse one SSE block into a validated event model.

    Returns ``(event, next_seq)``.  *event* is ``None`` when the block
    is a ``[DONE]`` sentinel, a keep-alive comment, or something we
    cannot validate (unknown type, bad JSON, etc.) — the caller decides
    how to handle those.
    """
    lines = block.split(b"\n")

    # Locate the single ``data:`` line.
    data_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(b"data:"):
            if data_idx is not None:
                return None, seq  # multiple data lines → skip
            data_idx = i

    if data_idx is None:
        return None, seq  # no data line (comment / keep-alive)

    raw = lines[data_idx][5:]  # strip "data:"
    if raw.startswith(b" "):
        raw = raw[1:]  # strip optional leading space

    if raw.strip() == b"[DONE]":
        return None, seq

    try:
        obj: dict = json.loads(raw)
    except json.JSONDecodeError:
        return None, seq

    if "type" not in obj:
        return None, seq

    had_seq = "sequence_number" in obj

    try:
        event = _stream_event_adapter.validate_python(obj)
    except ValidationError:
        return None, seq  # unknown or malformed event type

    if not had_seq:
        event.sequence_number = seq
    seq += 1

    return event, seq


async def _parse_response_stream(
    resp: httpx.Response,
) -> AsyncIterator[ResponseStreamEvent]:
    """Yield typed event models from an upstream SSE byte stream.

    Each ``data:`` payload is validated through the
    ``ResponseStreamEvent`` discriminated union so that missing fields
    are filled with spec-conformant defaults.  Non-event lines (comments,
    ``[DONE]``) are silently dropped — the route adds its own ``[DONE]``
    after the iterator is exhausted.

    The *resp* is closed when the generator finishes or is cancelled.
    """
    seq = 0
    buf = b""
    try:
        async for chunk in resp.aiter_bytes():
            buf += chunk
            while b"\n\n" in buf:
                event_block, buf = buf.split(b"\n\n", 1)
                event, seq = _parse_sse_event(event_block, seq)
                if event is not None:
                    yield event
        if buf.strip():
            event, seq = _parse_sse_event(buf, seq)
            if event is not None:
                yield event
    finally:
        await resp.aclose()


async def _create_event_stream(
    client: httpx.AsyncClient, url: str, body: CreateResponseRequest,
) -> AsyncIterator[ResponseStreamEvent]:
    """Open a streaming request and yield validated event models.

    Combines ``stream_raw`` (HTTP transport) with
    ``_parse_response_stream`` (SSE parsing + validation) into a single
    async generator.
    """
    resp = await stream_raw(client, url, body)
    async for event in _parse_response_stream(resp):
        yield event


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------


class LlamacppClient(
    BaseProvider,
    ChatCompletions,
    Completions,
    Responses,
    Embeddings,
    ModelManagement,
):
    name = "llamacpp"
    url_env = "LLAMACPP_URL"
    default_url = "http://llamacpp:8000"
    # Sensible fallback for llama-server presets when we can't determine
    # capabilities from the response (e.g. unloaded preset that
    # llama-server hasn't introspected yet). Most GGUF models support
    # completion; embeddings / multimodal are detected at load time.
    default_model_capabilities: ClassVar[list[str]] = ["completion"]
    models_path = "/v1/models"

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # -- Model discovery ------------------------------------------------------

    async def fetch_models(self) -> list[ModelObject]:
        """Return models by reading llama-server's native model listing.

        In router mode, llama-server's ``/v1/models`` response includes
        two parallel arrays:

        - ``data`` — OpenAI-shaped entries that now carry a ``status``
          object (``{"value": "loaded" | "loading" | "unloaded" |
          "sleeping", "args": [...]}``) for every known preset.
        - ``models`` — Ollama-shaped entries whose ``capabilities`` field
          lists the real per-model capabilities (``completion``,
          ``multimodal``, ``embeddings``, ...).

        We read ``data`` as the canonical list (so unloaded presets show
        up too) and merge capabilities from ``models`` when available.
        """
        url = self._url(self.models_path)
        try:
            resp = await self._http_client.get(url, timeout=10.0)
        except Exception:
            logger.warning(
                "llamacpp model discovery failed (%s) — may not be ready yet",
                url, exc_info=True,
            )
            return []

        if resp.status_code != 200:
            logger.warning(
                "llamacpp model discovery failed (%s): HTTP %d",
                url, resp.status_code,
            )
            return []

        payload = resp.json()
        data_entries = payload.get("data") or []
        ollama_entries = payload.get("models") or []

        # Build a capabilities lookup from the Ollama-shaped list.
        caps_by_id: dict[str, list[str]] = {}
        for entry in ollama_entries:
            model_id = entry.get("model") or entry.get("name") or ""
            if not model_id:
                continue
            caps = list(entry.get("capabilities") or [])
            if caps:
                caps_by_id[model_id] = caps

        result: list[ModelObject] = []
        seen: set[str] = set()
        for entry in data_entries:
            model_id = entry.get("id") or ""
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)

            status = _parse_llamacpp_status(entry.get("status"))
            capabilities = caps_by_id.get(
                model_id, list(self.default_model_capabilities),
            )
            result.append(
                ModelObject(
                    id=model_id,
                    created=int(entry.get("created") or 0),
                    owned_by=self.name,
                    capabilities=capabilities,
                    status=status,
                ),
            )

        # In non-router mode (single-model), the response has no
        # ``data`` entries we haven't already covered — but if it does,
        # or if only ``models`` came back, surface those too.
        for model_id, caps in caps_by_id.items():
            if model_id in seen:
                continue
            seen.add(model_id)
            result.append(
                ModelObject(
                    id=model_id,
                    owned_by=self.name,
                    capabilities=caps,
                    status=ModelStatus(value="loaded"),
                ),
            )

        return result

    # -- ModelManagement ------------------------------------------------------

    async def load_model(self, model: str) -> LoadModelResponse:
        """POST ``/models/load`` on the router (note: no ``/v1/`` prefix)."""
        return await self._model_action("/models/load", model)

    async def unload_model(self, model: str) -> UnloadModelResponse:
        """POST ``/models/unload`` on the router (note: no ``/v1/`` prefix)."""
        result = await self._model_action("/models/unload", model)
        return UnloadModelResponse(
            success=result.success,
            model=result.model,
            status=result.status,
        )

    async def _model_action(
        self, path: str, model: str,
    ) -> LoadModelResponse:
        """Shared POST helper for llama-server's load/unload endpoints."""
        resp = await self._http_client.post(
            self._url(path), json={"model": model},
        )
        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        payload = resp.json() if resp.content else {}
        return LoadModelResponse(
            success=bool(payload.get("success", True)),
            model=model,
            status=_parse_llamacpp_status(payload.get("status")),
        )

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

    def create_response_stream(
        self, body: CreateResponseRequest,
    ) -> AsyncIterator[ResponseStreamEvent]:
        return _create_event_stream(self._http_client, self._url("/v1/responses"), body)

    # -- Embeddings -----------------------------------------------------------

    async def embeddings(
        self, body: EmbeddingRequest,
    ) -> EmbeddingResponse:
        data = await forward(self._http_client, self._url("/v1/embeddings"), body)
        return EmbeddingResponse.model_validate(data)

"""JSON proxy transport helpers.

Utility functions for providers that transparently forward Pydantic
request models as JSON to a backend HTTP service.  Each function
takes an ``httpx.AsyncClient``, a full URL, and a Pydantic body.

Providers with non-JSON transports (multipart form data, custom
payloads) should use the ``httpx.AsyncClient`` directly instead —
see ``ChatterboxClient`` and ``WhisperxClient`` for examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from starlette.responses import StreamingResponse

if TYPE_CHECKING:
    import httpx
    from pydantic import BaseModel


async def forward(
    client: httpx.AsyncClient, url: str, body: BaseModel,
) -> dict[str, Any]:
    """POST *body* as JSON to *url* and return the parsed JSON response.

    Raises ``HTTPException`` if the backend returns a non-200 status.
    """
    payload = body.model_dump(exclude_unset=True)
    resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        detail = resp.text or f"Backend returned HTTP {resp.status_code}"
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


async def forward_stream(
    client: httpx.AsyncClient, url: str, body: BaseModel,
) -> StreamingResponse:
    """POST *body* as JSON to *url* and return a ``StreamingResponse``.

    The backend's ``Content-Type`` is forwarded (defaults to
    ``text/event-stream``).  The underlying httpx response is closed
    automatically when the stream finishes or the client disconnects.

    Raises ``HTTPException`` if the backend returns a non-200 status.
    """
    payload = body.model_dump(exclude_unset=True)
    req = client.build_request("POST", url, json=payload)
    resp = await client.send(req, stream=True)

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


async def stream_raw(
    client: httpx.AsyncClient, url: str, body: BaseModel,
) -> httpx.Response:
    """POST *body* as JSON to *url* and return the raw httpx response.

    Unlike ``forward_stream`` this returns the raw ``httpx.Response``
    instead of wrapping it in a ``StreamingResponse``, which is useful
    when the caller needs to process the stream itself (e.g. to
    interleave TTS audio into a chat completion stream).

    **The caller is responsible for closing the response** via
    ``await resp.aclose()``.

    Raises ``HTTPException`` if the backend returns a non-200 status.
    """
    payload = body.model_dump(exclude_unset=True)
    req = client.build_request("POST", url, json=payload)
    resp = await client.send(req, stream=True)

    if resp.status_code != 200:
        await resp.aread()
        detail = resp.text or f"Backend returned HTTP {resp.status_code}"
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return resp

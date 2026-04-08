"""Base class for HTTP provider clients.

Provides model discovery, health checking, and a create classmethod
for uniform instantiation.  Each subclass declares *url_env* (the
environment variable that holds the backend URL) and *default_url*
(the fallback when the variable is not set).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from gateway.models import ModelObject, ModelStatus

if TYPE_CHECKING:
    from typing import Self

    import httpx

logger = logging.getLogger(__name__)


def status_from_payload(raw: object) -> ModelStatus | None:
    """Convert a service-side ``status`` dict into a ``ModelStatus``.

    Accepts ``None`` or a dict; any other input (or a dict that fails
    validation) yields ``None`` so callers can substitute their own
    default. Shared helper used by every provider that forwards the
    new per-model status field from its backend.
    """
    if not isinstance(raw, dict):
        return None
    try:
        return ModelStatus.model_validate(raw)
    except Exception:
        logger.debug("Ignoring malformed status payload: %r", raw)
        return None


class BaseProvider:
    """Shared implementation for HTTP-based provider clients.

    Subclasses set name, url_env, default_url, and
    default_model_capabilities.
    """

    name: str
    url_env: ClassVar[str]
    default_url: ClassVar[str]
    default_model_capabilities: ClassVar[list[str]]
    models_path: str = "/models"

    def __init__(self, *, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    @classmethod
    def create(cls, http_client: httpx.AsyncClient) -> Self:
        url = os.environ.get(cls.url_env, cls.default_url)
        return cls(base_url=url, http_client=http_client)

    @property
    def base_url(self) -> str:
        return self._base_url

    async def fetch_models(self) -> list[ModelObject]:
        """Fetch models from the backend, injecting default capabilities.

        If the backend reports a ``status`` object per model (as
        chatterbox and whisperx now do, mirroring llama-server's
        shape), it's propagated onto ``ModelObject.status`` and used
        to derive the legacy ``loaded`` flag.
        """
        url = f"{self._base_url}{self.models_path}"
        try:
            resp = await self._http_client.get(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(
                    "Model discovery failed for %s (%s): HTTP %d",
                    self.name, url, resp.status_code,
                )
                return []
            data = resp.json()
            result: list[ModelObject] = []
            for m in data.get("data", []):
                status_raw = m.get("status")
                status: ModelStatus | None = None
                if isinstance(status_raw, dict):
                    try:
                        status = ModelStatus.model_validate(status_raw)
                    except Exception:
                        logger.debug(
                            "Ignoring malformed status for %s/%s",
                            self.name, m.get("id", ""),
                        )
                        status = None
                obj = ModelObject(
                    id=m.get("id", ""),
                    object=m.get("object", "model"),
                    created=m.get("created", 0),
                    # Always attribute models to this provider's name
                    # so the gateway's load/unload routes can dispatch
                    # by ``owned_by``. Service-side branding (e.g.
                    # "resemble-ai") is intentionally not surfaced.
                    owned_by=self.name,
                    capabilities=m.get("capabilities", []),
                    status=status,
                    loaded=m.get("loaded", True),
                )
                if not obj.capabilities:
                    obj.capabilities = list(self.default_model_capabilities)
                result.append(obj)
            return result
        except Exception:
            logger.warning(
                "Model discovery failed for %s (%s) — may not be ready yet",
                self.name, url, exc_info=True,
            )
            return []

    async def check_health(self) -> dict:
        url = f"{self._base_url}/health"
        try:
            resp = await self._http_client.get(url, timeout=5.0)
            if resp.status_code == 200:
                return {"status": "healthy"}
            return {"status": "unhealthy", "detail": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"status": "unreachable", "detail": f"{type(exc).__name__}: {exc}"}

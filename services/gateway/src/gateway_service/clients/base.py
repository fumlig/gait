"""Base class for HTTP backend service clients.

Provides shared model discovery and health checking.  Concrete clients
extend this class and implement one or more resource protocols from
:pymod:`gateway_service.clients.protocols`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from gateway_service.models import ModelObject

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class BaseBackend:
    """Shared implementation for HTTP-based backend clients.

    Subclasses must set ``name``, ``env_var``, and
    ``default_model_capabilities``.  Override ``models_path`` if the
    backend uses a non-standard path (e.g. ``/v1/models``).

    Which resource protocols a client implements determines which
    ``app.state`` slots it fills — see :pymod:`~gateway_service.clients.protocols`.
    """

    name: str
    """Human-readable name used in logs and health responses."""

    env_var: str
    """Environment variable that holds this backend's URL."""

    default_model_capabilities: ClassVar[list[str]]
    """Model-level capabilities injected when a model doesn't self-report."""

    models_path: str = "/models"
    """Path to the model listing endpoint."""

    def __init__(self, *, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    @property
    def base_url(self) -> str:
        return self._base_url

    async def fetch_models(self) -> list[ModelObject]:
        """Fetch models from the backend, injecting default capabilities."""
        url = f"{self._base_url}{self.models_path}"
        try:
            resp = await self._http_client.get(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(
                    "Model discovery failed for %s (%s): HTTP %d",
                    self.name,
                    url,
                    resp.status_code,
                )
                return []
            data = resp.json()
            result: list[ModelObject] = []
            for m in data.get("data", []):
                obj = ModelObject(
                    id=m.get("id", ""),
                    object=m.get("object", "model"),
                    created=m.get("created", 0),
                    owned_by=m.get("owned_by", ""),
                    capabilities=m.get("capabilities", []),
                    loaded=m.get("loaded", True),
                )
                if not obj.capabilities:
                    obj.capabilities = list(self.default_model_capabilities)
                result.append(obj)
            return result
        except Exception:
            logger.warning(
                "Model discovery failed for %s (%s) — it may not be ready yet",
                self.name,
                url,
                exc_info=True,
            )
            return []

    async def check_health(self) -> dict:
        """Check GET /health on the backend."""
        url = f"{self._base_url}/health"
        try:
            resp = await self._http_client.get(url, timeout=5.0)
            if resp.status_code == 200:
                return {"status": "healthy"}
            return {"status": "unhealthy", "detail": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"status": "unreachable", "detail": f"{type(exc).__name__}: {exc}"}

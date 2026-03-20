"""Gateway service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway configuration.

    All variables map directly from the environment (no prefix).

    Backend service URLs and the voices directory are **not** listed here.
    They are discovered via each client class's ``env_var`` attribute
    (see :pymod:`gateway_service.backends`).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Timeouts for backend requests (seconds)
    backend_timeout: float = 300.0  # ML inference can be slow


settings = Settings()

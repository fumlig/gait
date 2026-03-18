"""Gateway service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class BackendService(BaseSettings):
    """Configuration for a single backend service."""

    name: str
    url: str
    prefix: str  # path prefix that routes to this backend


class Settings(BaseSettings):
    """Gateway configuration.

    All variables map directly from the environment (no prefix).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Backend service URLs (internal Docker network addresses)
    chatterbox_url: str = "http://chatterbox:8000"
    whisperx_url: str = "http://whisperx:8000"

    # Timeouts for proxied requests (seconds)
    proxy_timeout: float = 300.0  # ML inference can be slow

    @property
    def backends(self) -> list[BackendService]:
        """Return the list of configured backend services."""
        return [
            BackendService(
                name="chatterbox",
                url=self.chatterbox_url,
                prefix="/v1/audio/speech",
            ),
            BackendService(
                name="whisperx",
                url=self.whisperx_url,
                prefix="/v1/audio/transcriptions",
            ),
            BackendService(
                name="whisperx",
                url=self.whisperx_url,
                prefix="/v1/audio/translations",
            ),
        ]


settings = Settings()

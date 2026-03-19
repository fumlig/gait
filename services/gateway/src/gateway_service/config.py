"""Gateway service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway configuration.

    All variables map directly from the environment (no prefix).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Backend service URLs (internal Docker network addresses)
    speech_url: str = "http://chatterbox:8000"
    transcription_url: str = "http://whisperx:8000"
    voice_url: str = "http://voice:8000"

    # Timeouts for backend requests (seconds)
    backend_timeout: float = 300.0  # ML inference can be slow


settings = Settings()

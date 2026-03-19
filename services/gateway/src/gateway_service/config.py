"""Gateway service configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

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
    chat_url: str = ""  # empty = chat backend disabled (e.g. http://llamacpp:8000)

    # Voice management (local filesystem, shared volume with chatterbox)
    voices_dir: Path = Path("/app/voices")

    # Timeouts for backend requests (seconds)
    backend_timeout: float = 300.0  # ML inference can be slow


settings = Settings()

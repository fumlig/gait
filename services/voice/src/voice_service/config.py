"""Voice service configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Voice service settings.

    All variables map directly from the environment (no prefix).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    host: str = "0.0.0.0"
    port: int = 8000

    # Directory containing named voice reference .wav files
    voices_dir: Path = Path("/app/voices")


settings = Settings()

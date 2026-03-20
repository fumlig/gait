from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Chatterbox TTS service configuration.

    All values can be overridden via environment variables
    (prefix-free, case-insensitive).
    """

    # Inference device: "cuda", "cpu", or "mps"
    device: str = "cuda"

    # Server bind address
    host: str = "0.0.0.0"
    port: int = 8000

    # Which model to preload at startup (empty string = lazy load all)
    default_model: str = ""

    # Directory containing named voice reference .wav files
    voices_dir: Path = Path("/app/voices")

    # HuggingFace model cache directory (mounted from host)
    model_cache_dir: Path = Path("/root/.cache/huggingface")

    # Idle timeout: unload models after N seconds of inactivity (0 = disabled)
    model_idle_timeout: int = 0

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()

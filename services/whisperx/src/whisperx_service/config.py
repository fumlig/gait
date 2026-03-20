"""Service configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """WhisperX service settings.

    All variables map directly from the environment (no prefix).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"

    # Model defaults (empty string = lazy load all)
    default_model: str = ""
    compute_type: str = "float16"
    batch_size: int = 16

    # Diarization (requires HF_TOKEN for pyannote gated models)
    enable_diarization: bool = False

    # Paths
    model_cache_dir: Path = Path("/root/.cache/huggingface")

    # Limits
    max_file_size: int = 25 * 1024 * 1024  # 25 MB (OpenAI limit)

    # Idle timeout: unload models after N seconds of inactivity (0 = disabled)
    model_idle_timeout: int = 0


settings = Settings()

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "case_sensitive": False}

    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    default_model: str = ""
    voices_dir: Path = Path("/app/voices")
    model_cache_dir: Path = Path("/root/.cache/huggingface")
    model_idle_timeout: int = 0  # seconds, 0 = disabled


settings = Settings()

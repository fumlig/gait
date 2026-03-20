from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "case_sensitive": False}

    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    default_model: str = ""
    compute_type: str = "float16"
    batch_size: int = 16
    enable_diarization: bool = False
    model_cache_dir: Path = Path("/root/.cache/huggingface")
    max_file_size: int = 25 * 1024 * 1024  # 25 MB
    model_idle_timeout: int = 0  # seconds, 0 = disabled


settings = Settings()

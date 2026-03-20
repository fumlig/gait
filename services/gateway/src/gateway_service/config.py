from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway configuration.

    Backend URLs and VOICES_DIR are discovered via each client class's
    env_var attribute (see backends/).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    host: str = "0.0.0.0"
    port: int = 8000
    backend_timeout: float = 300.0


settings = Settings()

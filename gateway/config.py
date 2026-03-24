from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway configuration.

    Set PROVIDERS to a comma-separated list of provider names to enable
    (e.g. "llamacpp,chatterbox,whisperx,voices"). Each provider reads
    its own env vars for configuration (URLs, paths, etc.).
    """

    model_config = {"env_prefix": "", "case_sensitive": False}

    host: str = "0.0.0.0"
    port: int = 8000
    backend_timeout: float = 300.0
    providers: str = "llamacpp,chatterbox,whisperx,voices"


settings = Settings()

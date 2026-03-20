"""Backend client registry.

Each backend class declares an env_var and a create classmethod. At startup
the gateway checks os.environ for each class's env var and calls create to
instantiate only those whose variable is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from gateway_service.backends.chatterbox import ChatterboxClient
from gateway_service.backends.llamacpp import LlamacppClient
from gateway_service.backends.voice import VoiceClient
from gateway_service.backends.whisperx import WhisperxClient

if TYPE_CHECKING:
    import httpx

__all__ = [
    "KNOWN_BACKENDS",
    "ChatterboxClient",
    "LlamacppClient",
    "Registerable",
    "VoiceClient",
    "WhisperxClient",
]


@runtime_checkable
class Registerable(Protocol):
    name: str
    env_var: str

    @classmethod
    def create(cls, env_value: str, http_client: httpx.AsyncClient) -> Registerable: ...


KNOWN_BACKENDS: list[type[Registerable]] = [
    LlamacppClient,
    ChatterboxClient,
    WhisperxClient,
    VoiceClient,
]

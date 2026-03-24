"""Provider client registry.

The gateway reads the PROVIDERS setting (a comma-separated list of
provider names) and instantiates only the matching classes from
KNOWN_PROVIDERS.  Each provider's create() classmethod reads its own
env vars for configuration (URLs, paths, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from gateway.providers.chatterbox import ChatterboxClient
from gateway.providers.llamacpp import LlamacppClient
from gateway.providers.voice import VoiceClient
from gateway.providers.whisperx import WhisperxClient

if TYPE_CHECKING:
    import httpx

__all__ = [
    "KNOWN_PROVIDERS",
    "ChatterboxClient",
    "LlamacppClient",
    "Registerable",
    "VoiceClient",
    "WhisperxClient",
]


@runtime_checkable
class Registerable(Protocol):
    name: str

    @classmethod
    def create(cls, http_client: httpx.AsyncClient) -> Registerable: ...


KNOWN_PROVIDERS: dict[str, type[Registerable]] = {
    "llamacpp": LlamacppClient,
    "chatterbox": ChatterboxClient,
    "whisperx": WhisperxClient,
    "voices": VoiceClient,
}

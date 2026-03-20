"""Client modules for communicating with backend services.

Each backend client class declares an ``env_var`` attribute.  At startup the
gateway checks ``os.environ`` for each known client's env var and instantiates
only those whose variable is set.

Which ``app.state`` slots a client fills is determined by the resource
protocols it implements (see :pymod:`~gateway_service.clients.protocols`),
discovered automatically via ``isinstance`` checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway_service.clients.chatterbox import ChatterboxClient
from gateway_service.clients.llamacpp import LlamacppClient
from gateway_service.clients.protocols import PROTOCOL_SLOTS
from gateway_service.clients.whisperx import WhisperxClient

if TYPE_CHECKING:
    from gateway_service.clients.base import BaseBackend

__all__ = [
    "KNOWN_BACKENDS",
    "PROTOCOL_SLOTS",
    "ChatterboxClient",
    "LlamacppClient",
    "WhisperxClient",
]

# Ordered list of all known HTTP backend client classes.
# The gateway iterates this list at startup and instantiates each client
# whose ``env_var`` is present in the environment.
KNOWN_BACKENDS: list[type[BaseBackend]] = [
    LlamacppClient,
    ChatterboxClient,
    WhisperxClient,
]

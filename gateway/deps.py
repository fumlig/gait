"""FastAPI dependencies for provider client injection.

Provides Annotated types that resolve provider clients from app.state,
raising HTTP 503 if the required backend is not configured.

Also provides a ``backend_errors`` context manager for consistent
error wrapping (HTTPException passthrough, other exceptions → 502).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import httpx
from fastapi import Depends, HTTPException, Request

from gateway.providers.protocols import (
    AudioSpeech,
    AudioTranscriptions,
    AudioTranslations,
    AudioVoices,
    ChatCompletions,
    Completions,
    Embeddings,
    Responses,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic factory
# ---------------------------------------------------------------------------


def _require(slot: str, label: str):
    """Return a FastAPI dependency that fetches a client from app.state.

    The returned function can also be called directly (e.g. inside helper
    functions that receive a ``Request`` but aren't route handlers).
    """

    def dependency(request: Request) -> object:
        client = getattr(request.app.state, slot, None)
        if client is None:
            raise HTTPException(status_code=503, detail=f"No {label} configured.")
        return client

    return dependency


# ---------------------------------------------------------------------------
# Dependency functions — usable with Depends() *or* called directly
# ---------------------------------------------------------------------------

require_chat = _require("chat_completions", "chat backend")
require_completions = _require("completions", "completions backend")
require_responses = _require("responses", "responses backend")
require_embeddings = _require("embeddings", "embeddings backend")
require_speech = _require("audio_speech", "speech backend")
require_transcription = _require("audio_transcriptions", "transcription backend")
require_translation = _require("audio_translations", "translation backend")
require_voices = _require("audio_voices", "voice management")


# ---------------------------------------------------------------------------
# Annotated types for route parameter injection
# ---------------------------------------------------------------------------

ChatClient = Annotated[ChatCompletions, Depends(require_chat)]
CompletionsClient = Annotated[Completions, Depends(require_completions)]
ResponsesClient = Annotated[Responses, Depends(require_responses)]
EmbeddingsClient = Annotated[Embeddings, Depends(require_embeddings)]
SpeechClient = Annotated[AudioSpeech, Depends(require_speech)]
TranscriptionClient = Annotated[AudioTranscriptions, Depends(require_transcription)]
TranslationClient = Annotated[AudioTranslations, Depends(require_translation)]
VoicesClient = Annotated[AudioVoices, Depends(require_voices)]


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


@asynccontextmanager
async def backend_errors(label: str) -> AsyncIterator[None]:
    """Wrap backend calls: pass through HTTPException, convert others to 502."""
    try:
        yield
    except HTTPException:
        raise
    except httpx.RemoteProtocolError:
        logger.warning("%s: backend disconnected (crashed or restarting)", label)
        raise HTTPException(
            status_code=503,
            detail=f"{label}: backend disconnected (crashed or restarting). Retry shortly.",
        )
    except Exception as exc:
        logger.exception("%s failed", label)
        raise HTTPException(status_code=502, detail=f"{label} unavailable.") from exc

"""FastAPI application entry point for the API gateway."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway_service.clients.speech import SpeechClient
from gateway_service.clients.transcription import TranscriptionClient
from gateway_service.clients.voice import VoiceClient
from gateway_service.config import settings
from gateway_service.models import ModelObject
from gateway_service.routes import health, models
from gateway_service.routes.audio import router as audio_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _fetch_models(client: httpx.AsyncClient, name: str, url: str) -> list[ModelObject]:
    """Fetch models from a backend service, logging and returning [] on failure."""
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code != 200:
            logger.warning(
                "Model discovery failed for %s (%s): HTTP %d", name, url, resp.status_code
            )
            return []
        data = resp.json()
        return [ModelObject(**m) for m in data.get("data", [])]
    except Exception:
        logger.warning(
            "Model discovery failed for %s (%s) — it may not be ready yet",
            name,
            url,
            exc_info=True,
        )
        return []


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create httpx client, build clients, discover models."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        follow_redirects=False,
    )

    # Create typed clients
    speech_client = SpeechClient(base_url=settings.speech_url, client=client)
    transcription_client = TranscriptionClient(base_url=settings.transcription_url, client=client)
    voice_client = VoiceClient(base_url=settings.voice_url, client=client)

    # Discover models from speech and transcription backends (concurrently)
    speech_models, transcription_models = await asyncio.gather(
        _fetch_models(client, "speech", f"{settings.speech_url.rstrip('/')}/models"),
        _fetch_models(client, "transcription", f"{settings.transcription_url.rstrip('/')}/models"),
    )

    # Merge and deduplicate
    seen: set[str] = set()
    merged: list[ModelObject] = []
    for model in [*speech_models, *transcription_models]:
        if model.id not in seen:
            seen.add(model.id)
            merged.append(model)

    logger.info("Model discovery complete — %d model(s)", len(merged))

    # Store on app state
    application.state.speech_client = speech_client
    application.state.transcription_client = transcription_client
    application.state.voice_client = voice_client
    application.state.models = merged

    logger.info(
        "Gateway started — backends: speech=%s, transcription=%s, voice=%s",
        settings.speech_url,
        settings.transcription_url,
        settings.voice_url,
    )
    yield

    await client.aclose()
    logger.info("Gateway stopped — httpx client closed")


app = FastAPI(
    title="Trave API Gateway",
    description="Unified OpenAI-compatible API gateway for local ML services",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS for local usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_router)
app.include_router(models.router)
app.include_router(health.router)

"""FastAPI application entry point for the API gateway."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway_service.clients.chat import ChatClient
from gateway_service.clients.speech import SpeechClient
from gateway_service.clients.transcription import TranscriptionClient
from gateway_service.clients.voice import VoiceClient
from gateway_service.config import settings
from gateway_service.models import ModelObject
from gateway_service.routes import completions, embeddings, health, models, responses
from gateway_service.routes.audio import router as audio_router
from gateway_service.routes.chat import router as chat_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Backend capabilities — backends don't return capabilities in their /models
# responses, so the gateway injects them based on which backend the model
# came from.
_CHAT_CAPABILITIES = ["chat", "completions", "embeddings"]
_SPEECH_CAPABILITIES = ["speech"]
_TRANSCRIPTION_CAPABILITIES = ["transcription"]


async def _fetch_models(
    client: httpx.AsyncClient, name: str, url: str
) -> list[ModelObject]:
    """Fetch models from a backend service, logging and returning [] on failure."""
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code != 200:
            logger.warning(
                "Model discovery failed for %s (%s): HTTP %d", name, url, resp.status_code
            )
            return []
        data = resp.json()
        raw_models = data.get("data", [])
        result = []
        for m in raw_models:
            # Parse with defaults — backends may or may not include new fields
            obj = ModelObject(
                id=m.get("id", ""),
                object=m.get("object", "model"),
                created=m.get("created", 0),
                owned_by=m.get("owned_by", ""),
                capabilities=m.get("capabilities", []),
                loaded=m.get("loaded", True),
            )
            # Inject capabilities based on which backend the model came from
            if not obj.capabilities:
                if name == "chat":
                    obj.capabilities = list(_CHAT_CAPABILITIES)
                elif name == "speech":
                    obj.capabilities = list(_SPEECH_CAPABILITIES)
                elif name == "transcription":
                    obj.capabilities = list(_TRANSCRIPTION_CAPABILITIES)
            result.append(obj)
        return result
    except Exception:
        logger.warning(
            "Model discovery failed for %s (%s) — it may not be ready yet",
            name,
            url,
            exc_info=True,
        )
        return []


async def fetch_all_models(application: FastAPI) -> list[ModelObject]:
    """Fetch models from all backends and merge (with deduplication).

    Also updates ``app.state.model_backends`` mapping.
    """
    client: httpx.AsyncClient = application.state._http_client

    speech_url = settings.speech_url
    transcription_url = settings.transcription_url
    chat_url = settings.chat_url

    futures = []
    backend_names = []

    if speech_url:
        futures.append(_fetch_models(client, "speech", f"{speech_url.rstrip('/')}/models"))
        backend_names.append("speech")
    if transcription_url:
        futures.append(
            _fetch_models(client, "transcription", f"{transcription_url.rstrip('/')}/models")
        )
        backend_names.append("transcription")
    if chat_url:
        futures.append(_fetch_models(client, "chat", f"{chat_url.rstrip('/')}/v1/models"))
        backend_names.append("chat")

    all_model_lists = await asyncio.gather(*futures)

    # Merge and deduplicate
    seen: set[str] = set()
    merged: list[ModelObject] = []

    for _backend_name, model_list in zip(backend_names, all_model_lists, strict=True):
        for model in model_list:
            if model.id not in seen:
                seen.add(model.id)
                merged.append(model)

    application.state.models = merged
    application.state.models_fetched_at = time.monotonic()

    return merged


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create httpx client, build clients, discover models."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        follow_redirects=False,
    )
    application.state._http_client = client

    # Create typed clients
    speech_client = SpeechClient(base_url=settings.speech_url, client=client)
    transcription_client = TranscriptionClient(base_url=settings.transcription_url, client=client)
    voice_client = VoiceClient(voices_dir=settings.voices_dir)

    chat_client: ChatClient | None = None
    if settings.chat_url:
        chat_client = ChatClient(base_url=settings.chat_url, client=client)

    # Store clients on app state
    application.state.speech_client = speech_client
    application.state.transcription_client = transcription_client
    application.state.voice_client = voice_client
    application.state.chat_client = chat_client

    # Initial model discovery
    merged = await fetch_all_models(application)
    logger.info("Model discovery complete — %d model(s)", len(merged))

    backends_msg = (
        f"speech={settings.speech_url}, transcription={settings.transcription_url}, "
        f"voices_dir={settings.voices_dir}"
    )
    if settings.chat_url:
        backends_msg += f", chat={settings.chat_url}"
    logger.info("Gateway started — backends: %s", backends_msg)
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

app.include_router(chat_router)
app.include_router(completions.router)
app.include_router(responses.router)
app.include_router(embeddings.router)
app.include_router(audio_router)
app.include_router(models.router)
app.include_router(health.router)

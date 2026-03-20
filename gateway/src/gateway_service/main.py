from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway_service.config import settings
from gateway_service.providers import KNOWN_PROVIDERS
from gateway_service.providers.base import BaseProvider
from gateway_service.providers.protocols import PROTOCOL_SLOTS
from gateway_service.routes import completions, embeddings, health, models, responses
from gateway_service.routes.audio import router as audio_router
from gateway_service.routes.chat import router as chat_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway_service.models import ModelObject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_all_models(application: FastAPI) -> list[ModelObject]:
    """Fetch models from all HTTP providers, merge, and deduplicate."""
    providers: list[BaseProvider] = getattr(application.state, "providers", [])

    all_model_lists = await asyncio.gather(
        *(provider.fetch_models() for provider in providers)
    )

    seen: set[str] = set()
    merged: list[ModelObject] = []
    for model_list in all_model_lists:
        for model in model_list:
            if model.id not in seen:
                seen.add(model.id)
                merged.append(model)

    application.state.models = merged
    application.state.models_fetched_at = time.monotonic()
    return merged


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Discover providers from env vars, create clients, fetch models."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        follow_redirects=False,
    )
    application.state._http_client = http_client

    # Instantiate providers whose env var is set
    all_clients: list[object] = []
    providers: list[BaseProvider] = []

    for cls in KNOWN_PROVIDERS:
        value = os.environ.get(cls.env_var)
        if not value:
            continue
        client = cls.create(value, http_client)
        all_clients.append(client)
        if isinstance(client, BaseProvider):
            providers.append(client)
        logger.info("Registered provider: %s → %s", cls.name, value)

    application.state.providers = providers

    # Wire resource protocols → app.state slots
    for client in all_clients:
        for protocol, slot in PROTOCOL_SLOTS:
            if isinstance(client, protocol):
                setattr(application.state, slot, client)
                logger.info("  %s → app.state.%s", type(client).__name__, slot)

    merged = await fetch_all_models(application)
    logger.info("Model discovery complete — %d model(s)", len(merged))

    provider_summary = ", ".join(f"{p.name}={p.base_url}" for p in providers)
    logger.info("Gateway started — providers: %s", provider_summary or "(none)")
    yield

    await http_client.aclose()
    logger.info("Gateway stopped — httpx client closed")


app = FastAPI(
    title="Gait API Gateway",
    description="Unified OpenAI-compatible API gateway for local ML services",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
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

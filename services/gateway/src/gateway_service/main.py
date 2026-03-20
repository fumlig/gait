"""FastAPI application entry point for the API gateway."""

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

from gateway_service.backends import KNOWN_BACKENDS
from gateway_service.backends.base import BaseBackend
from gateway_service.config import settings
from gateway_service.protocols import PROTOCOL_SLOTS
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
    """Fetch models from all HTTP backends and merge (with deduplication)."""
    backends: list[BaseBackend] = getattr(application.state, "backends", [])

    all_model_lists = await asyncio.gather(
        *(backend.fetch_models() for backend in backends)
    )

    # Merge and deduplicate (first occurrence wins)
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
    """Discover backends from env vars, create clients, fetch models."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_timeout, connect=10.0),
        follow_redirects=False,
    )
    application.state._http_client = http_client

    # ------------------------------------------------------------------
    # Instantiate all backends whose env var is set
    # ------------------------------------------------------------------
    all_clients: list[object] = []
    backends: list[BaseBackend] = []

    for cls in KNOWN_BACKENDS:
        value = os.environ.get(cls.env_var)
        if not value:
            continue
        client = cls.create(value, http_client)
        all_clients.append(client)
        if isinstance(client, BaseBackend):
            backends.append(client)
        logger.info("Registered backend: %s → %s", cls.name, value)

    application.state.backends = backends

    # ------------------------------------------------------------------
    # Wire resource protocols → app.state slots via isinstance checks
    # ------------------------------------------------------------------
    for client in all_clients:
        for protocol, slot in PROTOCOL_SLOTS:
            if isinstance(client, protocol):
                setattr(application.state, slot, client)
                logger.info("  %s → app.state.%s", type(client).__name__, slot)

    # ------------------------------------------------------------------
    # Initial model discovery
    # ------------------------------------------------------------------
    merged = await fetch_all_models(application)
    logger.info("Model discovery complete — %d model(s)", len(merged))

    backend_summary = ", ".join(f"{b.name}={b.base_url}" for b in backends)
    logger.info("Gateway started — backends: %s", backend_summary or "(none)")
    yield

    await http_client.aclose()
    logger.info("Gateway stopped — httpx client closed")


app = FastAPI(
    title="Trave API Gateway",
    description="Unified OpenAI-compatible API gateway for local ML services",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS for local usage
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

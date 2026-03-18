"""FastAPI application entry point for the Chatterbox TTS service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatterbox_service.config import settings
from chatterbox_service.engine import engine
from chatterbox_service.routes import health, models, speech

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Preload the default model on startup, release all on shutdown."""
    logger.info("Preloading default model: %s", settings.default_model)
    engine.load(settings.default_model)
    yield
    engine.unload()


app = FastAPI(
    title="Chatterbox TTS",
    description=(
        "OpenAI-compatible TTS API backed by Chatterbox models from Resemble AI. "
        "Supports chatterbox-turbo (350M), chatterbox (500M), and "
        "chatterbox-multilingual (500M, 23 languages). "
        "Models are loaded lazily on first request; the default model is preloaded at startup."
    ),
    version="0.2.0",
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

app.include_router(speech.router)
app.include_router(models.router)
app.include_router(health.router)

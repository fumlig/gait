"""FastAPI application entry point for the Chatterbox TTS service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Load the model on startup, release on shutdown."""
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="Chatterbox TTS",
    description="OpenAI-compatible TTS API backed by Chatterbox-Turbo",
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

app.include_router(speech.router)
app.include_router(models.router)
app.include_router(health.router)

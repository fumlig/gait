"""FastAPI application entry point for the WhisperX STT service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from whisperx_service.engine import engine
from whisperx_service.routes import health, models, transcriptions, translations

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the default model on startup, release on shutdown."""
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="WhisperX STT",
    description="OpenAI-compatible speech-to-text API backed by WhisperX",
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

app.include_router(transcriptions.router)
app.include_router(translations.router)
app.include_router(models.router)
app.include_router(health.router)

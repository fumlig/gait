"""FastAPI application entry point for the API gateway."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway_service.proxy import close_client, create_client
from gateway_service.routes import audio, health, models

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage the shared httpx client lifecycle."""
    create_client()
    logger.info("Gateway started — httpx client ready")
    yield
    await close_client()
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

app.include_router(audio.router)
app.include_router(models.router)
app.include_router(health.router)

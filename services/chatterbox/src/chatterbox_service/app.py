"""Minimal Chatterbox TTS service (Starlette).

RPC-style backend for the gateway.  No OpenAI-compatible routing —
just raw synthesize / models / health endpoints.

Endpoints:
    POST /synthesize  - JSON body -> audio/wav binary
    GET  /models      - list available models
    GET  /health      - liveness / readiness check
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import torch  # noqa: TC002 — used at runtime in _wav_to_bytes
import torchaudio
from starlette.applications import Starlette
from starlette.requests import Request  # noqa: TC002 — Starlette handler signature
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from chatterbox_service.config import settings
from chatterbox_service.engine import (
    KNOWN_MODELS,
    engine,
    validate_language,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    {"id": name, "object": "model", "owned_by": "resemble-ai"} for name in sorted(KNOWN_MODELS)
]


def _wav_to_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform tensor to WAV bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, wav, sample_rate, format="wav")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def synthesize(request: Request) -> Response:
    """Generate speech from text.  Returns raw WAV bytes."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    # Required fields
    text = body.get("text")
    voice = body.get("voice")
    model = body.get("model")
    if not text or not voice or not model:
        return JSONResponse(
            {"detail": "'text', 'voice', and 'model' are required fields."},
            status_code=400,
        )

    # Validate and ensure model is loaded
    try:
        model_name = engine.ensure_model(model)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # Validate language for multilingual models
    language = body.get("language")
    try:
        validate_language(model_name, language)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "No model is loaded yet."}, status_code=503)

    # Optional params with defaults matching SpeechRequest schema
    speed = float(body.get("speed", 1.0))
    exaggeration = float(body.get("exaggeration", 0.5))
    cfg_weight = float(body.get("cfg_weight", 0.5))
    temperature = float(body.get("temperature", 0.8))
    repetition_penalty = float(body.get("repetition_penalty", 1.2))
    top_p = float(body.get("top_p", 1.0))
    min_p = float(body.get("min_p", 0.05))
    top_k = int(body.get("top_k", 1000))
    seed = body.get("seed")
    if seed is not None:
        seed = int(seed)

    # Generate
    try:
        wav, sr = engine.generate(
            text=text,
            model_name=model_name,
            voice=voice,
            speed=speed,
            language=language,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            seed=seed,
        )
    except Exception:
        logger.exception("Speech generation failed")
        return JSONResponse({"detail": "Speech generation failed."}, status_code=500)

    audio_bytes = _wav_to_bytes(wav, sr)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )


async def list_models(request: Request) -> JSONResponse:
    """Return available model list."""
    return JSONResponse({"object": "list", "data": AVAILABLE_MODELS})


async def health(request: Request) -> JSONResponse:
    """Liveness / readiness check."""
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": engine.is_loaded,
            "models": engine.loaded_models,
        }
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Preload the default model on startup, release on shutdown."""
    logger.info("Preloading default model: %s", settings.default_model)
    engine.load(settings.default_model)
    yield
    engine.unload()


app = Starlette(
    routes=[
        Route("/synthesize", synthesize, methods=["POST"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)

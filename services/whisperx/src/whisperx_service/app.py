"""Minimal WhisperX STT service (Starlette).

RPC-style backend for the gateway.  No OpenAI-compatible routing —
just raw transcribe / translate / models / health endpoints.

Endpoints:
    POST /transcribe  - multipart (file + params) -> JSON segments
    POST /translate   - multipart (file + params) -> JSON segments
    GET  /models      - list available models
    GET  /health      - liveness / readiness check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.requests import Request  # noqa: TC002 — Starlette handler signature
from starlette.responses import JSONResponse
from starlette.routing import Route

from whisperx_service.config import settings
from whisperx_service.engine import engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared transcription logic
# ---------------------------------------------------------------------------


async def _do_transcribe(request: Request, *, task: str) -> JSONResponse:
    """Shared handler for transcription and translation.

    Reads multipart form data with ``file``, ``model``, and optional
    ``language``, ``prompt``, ``temperature``, ``word_timestamps`` fields.
    """
    form = await request.form()

    upload = form.get("file")
    model = form.get("model", "")

    if not model:
        return JSONResponse({"detail": "'model' is required."}, status_code=400)

    # Ensure model is loaded
    try:
        engine.ensure_model(str(model))
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "Model is not loaded yet."}, status_code=503)

    # Read and validate file
    if upload is None:
        return JSONResponse({"detail": "No audio file uploaded."}, status_code=400)

    audio_data = await upload.read()
    if not audio_data:
        return JSONResponse({"detail": "Empty audio file."}, status_code=400)

    if len(audio_data) > settings.max_file_size:
        return JSONResponse(
            {"detail": f"File too large. Maximum size is {settings.max_file_size} bytes."},
            status_code=400,
        )

    # Optional params
    language = form.get("language") or None
    prompt = form.get("prompt") or None
    temperature = float(form.get("temperature", 0.0))
    word_timestamps_raw = str(form.get("word_timestamps", "false"))
    want_words = word_timestamps_raw.lower() == "true"
    want_diarize = want_words and settings.enable_diarization

    # Build kwargs (only pass language for transcribe, not translate)
    transcribe_kwargs: dict = {
        "prompt": prompt,
        "temperature": temperature,
        "task": task,
        "word_timestamps": want_words,
        "diarize": want_diarize,
    }
    if task == "transcribe" and language:
        transcribe_kwargs["language"] = language

    # Run inference
    try:
        result = engine.transcribe(audio_data, **transcribe_kwargs)
    except Exception:
        label = "Transcription" if task == "transcribe" else "Translation"
        logger.exception("%s failed", label)
        return JSONResponse({"detail": f"{label} failed."}, status_code=500)

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def transcribe(request: Request) -> JSONResponse:
    """Transcribe audio to text (preserves source language)."""
    return await _do_transcribe(request, task="transcribe")


async def translate(request: Request) -> JSONResponse:
    """Translate audio to English text."""
    return await _do_transcribe(request, task="translate")


async def list_models(request: Request) -> JSONResponse:
    """Return available model list."""
    models = [
        {"id": model_id, "object": "model", "owned_by": "whisperx"}
        for model_id in engine.list_available_models()
    ]
    return JSONResponse({"object": "list", "data": models})


async def health(request: Request) -> JSONResponse:
    """Liveness / readiness check."""
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": engine.is_loaded,
        }
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Load the default model on startup, release on shutdown."""
    engine.load()
    yield
    engine.unload()


app = Starlette(
    routes=[
        Route("/transcribe", transcribe, methods=["POST"]),
        Route("/translate", translate, methods=["POST"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)

"""Minimal WhisperX STT service (Starlette).

RPC-style backend for the gateway.  No OpenAI-compatible routing —
just raw transcribe / translate / models / health endpoints.

Models are loaded lazily on first request via ``engine.ensure_model``
and unloaded automatically after a configurable idle timeout.

Endpoints:
    POST /transcribe  - multipart (file + params) -> JSON segments
    POST /translate   - multipart (file + params) -> JSON segments
    GET  /models      - list available models with loaded status
    GET  /health      - liveness / readiness check
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
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

    # Ensure model is loaded (lazy loading)
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

    # Diarization: per-request opt-in via `diarize` field, but only if the
    # server has diarization enabled in its config (ENABLE_DIARIZATION=true)
    # AND a HuggingFace token is available (required by pyannote).
    diarize_raw = str(form.get("diarize", "false"))
    want_diarize = (
        diarize_raw.lower() == "true"
        and settings.enable_diarization
        and bool(os.getenv("HF_TOKEN"))
    )

    # Diarization needs word-level alignment to assign speakers to words
    if want_diarize:
        want_words = True

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
    """Return available model list with capabilities and loaded status."""
    loaded_name = engine.loaded_model_name
    models = [
        {
            "id": model_id,
            "object": "model",
            "owned_by": "whisperx",
            "capabilities": ["transcription", "translation"],
            "loaded": model_id == loaded_name
            or (model_id == "whisper-1" and loaded_name is not None),
        }
        for model_id in engine.list_available_models()
    ]
    return JSONResponse({"object": "list", "data": models})


async def health(request: Request) -> JSONResponse:
    """Liveness / readiness check."""
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": engine.is_loaded,
            "loaded_model": engine.loaded_model_name,
        }
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Optionally preload the default model, start idle checker."""
    idle_task = None

    # Start idle timeout background task if configured
    if settings.model_idle_timeout > 0:

        async def _idle_checker() -> None:
            while True:
                await asyncio.sleep(30)
                if engine.unload_if_idle(settings.model_idle_timeout):
                    logger.info(
                        "Model unloaded due to idle timeout (%ds).",
                        settings.model_idle_timeout,
                    )

        idle_task = asyncio.create_task(_idle_checker())

    # Optionally preload default model
    if settings.default_model:
        logger.info("Preloading default model: %s", settings.default_model)
        engine.load()
    else:
        logger.info(
            "No default model configured — models will be loaded lazily on first request."
        )

    yield

    if idle_task:
        idle_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await idle_task
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

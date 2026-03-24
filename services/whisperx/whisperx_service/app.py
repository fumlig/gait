"""WhisperX STT service (Starlette).

Endpoints:
    POST /transcribe         - multipart (file + params) -> JSON segments
    POST /transcribe_stream  - multipart (file + params) -> SSE segments
    POST /translate          - multipart (file + params) -> JSON segments
    GET  /models             - available models with loaded status
    GET  /health             - liveness / readiness
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.requests import Request  # noqa: TC002 — Starlette handler signature
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from whisperx_service.config import settings
from whisperx_service.engine import engine
from whisperx_service.idle import idle_checker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _do_transcribe(request: Request, *, task: str) -> JSONResponse:
    """Shared handler for transcription and translation."""
    form = await request.form()

    upload = form.get("file")
    model = str(form.get("model", ""))

    if not model:
        return JSONResponse({"detail": "'model' is required."}, status_code=400)

    try:
        engine.ensure_model(model)
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "Model is not loaded yet."}, status_code=503)

    from starlette.datastructures import UploadFile

    if not isinstance(upload, UploadFile):
        return JSONResponse({"detail": "No audio file uploaded."}, status_code=400)

    audio_data = await upload.read()
    if not audio_data:
        return JSONResponse({"detail": "Empty audio file."}, status_code=400)

    if len(audio_data) > settings.max_file_size:
        return JSONResponse(
            {"detail": f"File too large. Maximum size is {settings.max_file_size} bytes."},
            status_code=400,
        )

    language = str(form.get("language", "")) or None
    prompt = str(form.get("prompt", "")) or None
    temperature = float(str(form.get("temperature", "0.0")))
    want_words = str(form.get("word_timestamps", "false")).lower() == "true"

    # Diarization: per-request opt-in, requires server config and HF_TOKEN
    want_diarize = (
        str(form.get("diarize", "false")).lower() == "true"
        and settings.enable_diarization
        and bool(os.getenv("HF_TOKEN"))
    )
    if want_diarize:
        want_words = True  # diarization needs word alignment

    try:
        result = engine.transcribe(
            audio_data,
            language=language if task == "transcribe" else None,
            prompt=prompt,
            temperature=temperature,
            task=task,
            word_timestamps=want_words,
            diarize=want_diarize,
        )
    except Exception:
        label = "Transcription" if task == "transcribe" else "Translation"
        logger.exception("%s failed", label)
        return JSONResponse({"detail": f"{label} failed."}, status_code=500)

    return JSONResponse(result)


async def transcribe(request: Request) -> JSONResponse:
    return await _do_transcribe(request, task="transcribe")


async def transcribe_stream(request: Request) -> StreamingResponse | JSONResponse:
    """POST /transcribe_stream — yield segments as SSE while the model runs."""
    form = await request.form()

    upload = form.get("file")
    model = str(form.get("model", ""))

    if not model:
        return JSONResponse({"detail": "'model' is required."}, status_code=400)

    try:
        engine.ensure_model(model)
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "Model is not loaded yet."}, status_code=503)

    from starlette.datastructures import UploadFile

    if not isinstance(upload, UploadFile):
        return JSONResponse({"detail": "No audio file uploaded."}, status_code=400)

    audio_data = await upload.read()
    if not audio_data:
        return JSONResponse({"detail": "Empty audio file."}, status_code=400)

    if len(audio_data) > settings.max_file_size:
        return JSONResponse(
            {"detail": f"File too large. Maximum size is {settings.max_file_size} bytes."},
            status_code=400,
        )

    language = str(form.get("language", "")) or None
    prompt = str(form.get("prompt", "")) or None
    temperature = float(str(form.get("temperature", "0.0")))

    def generate():
        """Sync generator — Starlette runs each next() in a thread."""
        for item in engine.transcribe_stream(
            audio_data,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task="transcribe",
        ):
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def translate(request: Request) -> JSONResponse:
    return await _do_transcribe(request, task="translate")


async def list_models(request: Request) -> JSONResponse:
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
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": engine.is_loaded,
            "loaded_model": engine.loaded_model_name,
        }
    )


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    if settings.default_model:
        logger.info("Preloading default model: %s", settings.default_model)
        engine.load()
    else:
        logger.info("No default model — models will load lazily on first request.")

    async with idle_checker(engine, settings.model_idle_timeout):
        yield

    engine.unload()


app = Starlette(
    routes=[
        Route("/transcribe", transcribe, methods=["POST"]),
        Route("/transcribe_stream", transcribe_stream, methods=["POST"]),
        Route("/translate", translate, methods=["POST"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)

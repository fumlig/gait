"""WhisperX STT service (Starlette).

Endpoints:
    POST /transcribe          - multipart (file + params) -> JSON segments
    POST /transcribe_stream   - multipart (file + params) -> SSE segments
    POST /translate           - multipart (file + params) -> JSON segments
    GET  /models              - available models with per-model status
    POST /models/load         - load a model (unloads any previous one)
    POST /models/unload       - unload the current model
    GET  /health              - liveness / readiness
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request  # noqa: TC002 — Starlette handler signature
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from whisperx_service.config import settings
from whisperx_service.engine import engine
from whisperx_service.idle import idle_checker
from whisperx_service.schemas import (
    HealthResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelListResponse,
    ModelStatus,
    TranscribeFormRequest,
    TranscribeStreamFormRequest,
    UnloadModelRequest,
    UnloadModelResponse,
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


def _validation_error(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.errors(include_url=False, include_context=False)},
        status_code=400,
    )


def _parse_form(
    form_mapping: dict[str, Any], model_cls: type[BaseModel],
) -> tuple[BaseModel | None, JSONResponse | None]:
    """Validate a form-field dict against *model_cls*.

    Returns ``(parsed_model, None)`` on success or
    ``(None, error_response)`` on validation failure.
    """
    try:
        parsed = model_cls.model_validate(form_mapping)
    except ValidationError as exc:
        return None, _validation_error(exc)
    return parsed, None


async def _parse_json_body(
    request: Request, model_cls: type[BaseModel],
) -> tuple[BaseModel | None, JSONResponse | None]:
    """Validate a JSON body against *model_cls*."""
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return None, JSONResponse(
            {"detail": "Invalid JSON body."}, status_code=400,
        )
    try:
        parsed = model_cls.model_validate(raw)
    except ValidationError as exc:
        return None, _validation_error(exc)
    return parsed, None


def _form_field_mapping(form) -> dict[str, Any]:
    """Convert a Starlette ``FormData`` into a plain dict for pydantic.

    Upload files are dropped; only scalar text fields are kept.
    """
    out: dict[str, Any] = {}
    for key in form.keys():  # noqa: SIM118 — Starlette form exposes .keys()
        value = form.get(key)
        if isinstance(value, UploadFile):
            continue
        if value is None or value == "":
            continue
        out[key] = value
    return out


def _model_status(model_name: str) -> ModelStatus:
    return ModelStatus(value=engine.status_for(model_name))  # ty: ignore[invalid-argument-type]


def _build_model_list() -> ModelListResponse:
    entries = [
        ModelInfo(
            id=model_id,
            capabilities=["transcription", "translation"],
            status=_model_status(model_id),
            loaded=engine.status_for(model_id) == "loaded",
        )
        for model_id in engine.list_available_models()
    ]
    return ModelListResponse(data=entries)


# ---------------------------------------------------------------------------
# Transcribe / translate
# ---------------------------------------------------------------------------


async def _do_transcribe(request: Request, *, task: str) -> JSONResponse:
    """Shared handler for transcription and translation."""
    form = await request.form()

    parsed, err = _parse_form(_form_field_mapping(form), TranscribeFormRequest)
    if err is not None:
        return err
    assert isinstance(parsed, TranscribeFormRequest)
    body = parsed

    try:
        engine.ensure_model(body.model)
    except Exception as exc:
        logger.exception("Failed to load model '%s'", body.model)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "Model is not loaded yet."}, status_code=503)

    upload = form.get("file")
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

    # Diarization: per-request opt-in, requires server config and HF_TOKEN
    want_diarize = (
        body.diarize
        and settings.enable_diarization
        and bool(os.getenv("HF_TOKEN"))
    )
    want_words = body.word_timestamps
    if want_diarize:
        want_words = True  # diarization needs word alignment

    try:
        result = engine.transcribe(
            audio_data,
            language=body.language if task == "transcribe" else None,
            prompt=body.prompt,
            temperature=body.temperature,
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

    parsed, err = _parse_form(
        _form_field_mapping(form), TranscribeStreamFormRequest,
    )
    if err is not None:
        return err
    assert isinstance(parsed, TranscribeStreamFormRequest)
    body = parsed

    try:
        engine.ensure_model(body.model)
    except Exception as exc:
        logger.exception("Failed to load model '%s'", body.model)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "Model is not loaded yet."}, status_code=503)

    upload = form.get("file")
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

    def generate():
        """Sync generator — Starlette runs each next() in a thread."""
        for item in engine.transcribe_stream(
            audio_data,
            language=body.language,
            prompt=body.prompt,
            temperature=body.temperature,
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


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


async def list_models(request: Request) -> JSONResponse:
    return JSONResponse(_build_model_list().model_dump())


async def load_model(request: Request) -> JSONResponse:
    parsed, err = await _parse_json_body(request, LoadModelRequest)
    if err is not None:
        return err
    assert isinstance(parsed, LoadModelRequest)

    try:
        resolved = engine.ensure_model(parsed.model)
    except Exception:
        logger.exception("Failed to load model '%s'", parsed.model)
        return JSONResponse(
            {"detail": f"Failed to load model '{parsed.model}'."},
            status_code=500,
        )

    response = LoadModelResponse(
        success=True,
        model=resolved,
        status=_model_status(resolved),
    )
    return JSONResponse(response.model_dump())


async def unload_model(request: Request) -> JSONResponse:
    parsed, err = await _parse_json_body(request, UnloadModelRequest)
    if err is not None:
        return err
    assert isinstance(parsed, UnloadModelRequest)

    # WhisperX keeps exactly one model resident at a time; any
    # ``model`` that matches the currently-loaded one (or is
    # ``whisper-1``) unloads it. Anything else is a no-op but still
    # succeeds, so the gateway's idempotent semantics hold.
    try:
        engine.unload()
    except Exception:
        logger.exception("Failed to unload model '%s'", parsed.model)
        return JSONResponse(
            {"detail": f"Failed to unload model '{parsed.model}'."},
            status_code=500,
        )

    response = UnloadModelResponse(
        success=True,
        model=parsed.model,
        status=_model_status(parsed.model),
    )
    return JSONResponse(response.model_dump())


# ---------------------------------------------------------------------------
# Health + lifespan
# ---------------------------------------------------------------------------


async def health(request: Request) -> JSONResponse:
    body = HealthResponse(
        model_loaded=engine.is_loaded,
        loaded_model=engine.loaded_model_name,
        phase=engine.status_phase,
    )
    return JSONResponse(body.model_dump())


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
        Route("/models/load", load_model, methods=["POST"]),
        Route("/models/unload", unload_model, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)

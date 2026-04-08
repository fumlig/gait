"""Chatterbox TTS service (Starlette).

Endpoints:
    POST /synthesize      - JSON body -> audio/wav
    GET  /models          - available models with per-model status
    POST /models/load     - load a model (unloads any previous one)
    POST /models/unload   - unload a specific model (or all)
    GET  /health          - liveness / readiness
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import torch  # noqa: TC002 — used at runtime in _wav_to_bytes
import torchaudio
from pydantic import BaseModel, ValidationError
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
from chatterbox_service.idle import idle_checker
from chatterbox_service.schemas import (
    HealthResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelListResponse,
    ModelStatus,
    SynthesizeRequest,
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


def _build_model_list() -> ModelListResponse:
    entries = [
        ModelInfo(
            id=name,
            owned_by="resemble-ai",
            capabilities=["speech"],
            status=ModelStatus(value=engine.status_for(name)),  # ty: ignore[invalid-argument-type]
            loaded=engine.loaded_models.get(name, False),
        )
        for name in sorted(KNOWN_MODELS)
    ]
    return ModelListResponse(data=entries)


def _model_status_response(model_name: str) -> ModelStatus:
    return ModelStatus(value=engine.status_for(model_name))  # ty: ignore[invalid-argument-type]


def _wav_to_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    torchaudio.save(buf, wav, sample_rate, format="wav")
    return buf.getvalue()


def _validation_error(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.errors(include_url=False, include_context=False)},
        status_code=400,
    )


async def _parse_json_body(
    request: Request, model_cls: type[BaseModel],
) -> tuple[BaseModel | None, JSONResponse | None]:
    """Return (parsed_model, None) on success or (None, error_response)."""
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def synthesize(request: Request) -> Response:
    parsed, err = await _parse_json_body(request, SynthesizeRequest)
    if err is not None:
        return err
    assert isinstance(parsed, SynthesizeRequest)
    body = parsed

    try:
        model_name = engine.ensure_model(body.model)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    try:
        validate_language(model_name, body.language)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if not engine.is_loaded:
        return JSONResponse({"detail": "No model is loaded yet."}, status_code=503)

    try:
        wav, sr = engine.generate(
            text=body.text,
            model_name=model_name,
            voice=body.voice,
            speed=body.speed,
            language=body.language,
            exaggeration=body.exaggeration,
            cfg_weight=body.cfg_weight,
            temperature=body.temperature,
            repetition_penalty=body.repetition_penalty,
            top_p=body.top_p,
            min_p=body.min_p,
            top_k=body.top_k,
            seed=body.seed,
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
    return JSONResponse(_build_model_list().model_dump())


async def load_model(request: Request) -> JSONResponse:
    parsed, err = await _parse_json_body(request, LoadModelRequest)
    if err is not None:
        return err
    assert isinstance(parsed, LoadModelRequest)

    try:
        resolved = engine.ensure_model(parsed.model)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Failed to load model '%s'", parsed.model)
        return JSONResponse(
            {"detail": f"Failed to load model '{parsed.model}'."},
            status_code=500,
        )

    response = LoadModelResponse(
        success=True,
        model=resolved,
        status=_model_status_response(resolved),
    )
    return JSONResponse(response.model_dump())


async def unload_model(request: Request) -> JSONResponse:
    parsed, err = await _parse_json_body(request, UnloadModelRequest)
    if err is not None:
        return err
    assert isinstance(parsed, UnloadModelRequest)

    model_name = parsed.model
    if model_name not in KNOWN_MODELS:
        return JSONResponse(
            {"detail": f"Unknown model '{model_name}'."}, status_code=404,
        )

    try:
        engine.unload(model_name)
    except Exception:
        logger.exception("Failed to unload model '%s'", model_name)
        return JSONResponse(
            {"detail": f"Failed to unload model '{model_name}'."},
            status_code=500,
        )

    response = UnloadModelResponse(
        success=True,
        model=model_name,
        status=_model_status_response(model_name),
    )
    return JSONResponse(response.model_dump())


async def health(request: Request) -> JSONResponse:
    body = HealthResponse(
        model_loaded=engine.is_loaded,
        models=engine.loaded_models,
        phase=engine.status_phase,
    )
    return JSONResponse(body.model_dump())


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    if settings.default_model:
        logger.info("Preloading default model: %s", settings.default_model)
        engine.load(settings.default_model)
    else:
        logger.info("No default model — models will load lazily on first request.")

    async with idle_checker(engine, settings.model_idle_timeout):
        yield

    engine.unload()


app = Starlette(
    routes=[
        Route("/synthesize", synthesize, methods=["POST"]),
        Route("/models", list_models, methods=["GET"]),
        Route("/models/load", load_model, methods=["POST"]),
        Route("/models/unload", unload_model, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)

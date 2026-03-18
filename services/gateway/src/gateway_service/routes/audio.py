"""Proxy routes for /v1/audio/* endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from gateway_service.config import settings
from gateway_service.proxy import httpx_to_fastapi_response, proxy_request

if TYPE_CHECKING:
    from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/audio/speech")
async def proxy_speech(request: Request) -> Response:
    """Forward TTS requests to the chatterbox backend."""
    try:
        resp = await proxy_request(settings.chatterbox_url, request)
    except Exception as exc:
        logger.exception("Failed to proxy to chatterbox")
        raise HTTPException(status_code=502, detail="Chatterbox backend unavailable.") from exc
    return httpx_to_fastapi_response(resp)


@router.post("/v1/audio/transcriptions")
async def proxy_transcriptions(request: Request) -> Response:
    """Forward STT transcription requests to the whisperx backend."""
    try:
        resp = await proxy_request(settings.whisperx_url, request)
    except Exception as exc:
        logger.exception("Failed to proxy to whisperx")
        raise HTTPException(status_code=502, detail="WhisperX backend unavailable.") from exc
    return httpx_to_fastapi_response(resp)


@router.post("/v1/audio/translations")
async def proxy_translations(request: Request) -> Response:
    """Forward STT translation requests to the whisperx backend."""
    try:
        resp = await proxy_request(settings.whisperx_url, request)
    except Exception as exc:
        logger.exception("Failed to proxy to whisperx")
        raise HTTPException(status_code=502, detail="WhisperX backend unavailable.") from exc
    return httpx_to_fastapi_response(resp)

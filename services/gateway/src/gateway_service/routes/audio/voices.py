"""Voice management routes — local filesystem operations via VoiceClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

if TYPE_CHECKING:
    from gateway_service.protocols import AudioVoices

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_voice_client(request: Request) -> AudioVoices:
    """Resolve the voice client from app state."""
    client = getattr(request.app.state, "audio_voices", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Voice management not configured.")
    return client


@router.get("/v1/audio/voices")
async def list_voices(request: Request) -> dict[str, object]:
    """List all available voices."""
    client = _get_voice_client(request)
    voices = await client.list_voices()
    return {
        "object": "list",
        "data": [v.model_dump() for v in voices],
    }


@router.get("/v1/audio/voices/{name}")
async def get_voice(name: str, request: Request) -> dict[str, str]:
    """Get a single voice by name."""
    client = _get_voice_client(request)
    voice = await client.get_voice(name)
    return voice.model_dump()


@router.post("/v1/audio/voices", status_code=201)
async def create_voice(
    request: Request,
    name: str = Form(..., description="Name for the new voice."),
    file: UploadFile = File(..., description="WAV audio sample for voice cloning."),
) -> dict[str, str]:
    """Upload a new voice reference clip."""
    client = _get_voice_client(request)

    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    voice = await client.create_voice(name, audio_data)
    return voice.model_dump()


@router.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str, request: Request) -> dict[str, object]:
    """Delete a voice by name."""
    client = _get_voice_client(request)
    return await client.delete_voice(name)

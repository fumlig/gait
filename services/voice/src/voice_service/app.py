"""Minimal voice management service (Starlette).

Manages WAV reference clips on a shared volume. No ML dependencies.
Endpoints:
    GET  /voices          - list all voices
    GET  /voices/{name}   - get a single voice
    POST /voices          - upload a new voice (multipart: name + file)
    DELETE /voices/{name} - delete a voice
    GET  /health          - liveness check
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from voice_service.config import settings

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.requests import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voices_dir() -> Path:
    return settings.voices_dir


# The "default" voice is a virtual entry that maps to the chatterbox
# library's built-in reference clip (no WAV file on disk).
_DEFAULT_VOICE = "default"


def _list_voice_names() -> list[str]:
    d = _voices_dir()
    disk_voices = sorted(p.stem for p in d.glob("*.wav")) if d.is_dir() else []
    # Always include "default" first.
    return [_DEFAULT_VOICE, *[n for n in disk_voices if n != _DEFAULT_VOICE]]


def _voice_obj(name: str) -> dict:
    return {"voice_id": name, "name": name}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def list_voices(request: Request) -> JSONResponse:
    names = _list_voice_names()
    return JSONResponse({"object": "list", "data": [_voice_obj(n) for n in names]})


async def get_voice(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    if name not in _list_voice_names():
        return JSONResponse({"detail": f"Voice '{name}' not found."}, status_code=404)
    return JSONResponse(_voice_obj(name))


async def create_voice(request: Request) -> JSONResponse:
    form = await request.form()
    name = form.get("name", "")
    upload = form.get("file")

    if not name or not _NAME_RE.match(name):
        return JSONResponse(
            {
                "detail": (
                    "Voice name must be non-empty and contain only"
                    " alphanumeric characters, hyphens, or underscores."
                )
            },
            status_code=400,
        )

    if name == _DEFAULT_VOICE:
        return JSONResponse(
            {"detail": "The 'default' voice is built-in and cannot be replaced."},
            status_code=400,
        )

    dest = _voices_dir() / f"{name}.wav"
    if dest.exists():
        return JSONResponse(
            {"detail": f"Voice '{name}' already exists."},
            status_code=409,
        )

    if upload is None:
        return JSONResponse({"detail": "No file uploaded."}, status_code=400)

    content = await upload.read()

    if len(content) < 44:
        return JSONResponse({"detail": "File too small to be a valid WAV file."}, status_code=400)

    if content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        return JSONResponse({"detail": "Uploaded file is not a valid WAV file."}, status_code=400)

    _voices_dir().mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    logger.info("Created voice '%s' (%d bytes)", name, len(content))
    return JSONResponse(_voice_obj(name), status_code=201)


async def delete_voice(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    if name == _DEFAULT_VOICE:
        return JSONResponse(
            {"detail": "The 'default' voice is built-in and cannot be deleted."},
            status_code=400,
        )
    path = _voices_dir() / f"{name}.wav"
    if not path.is_file():
        return JSONResponse({"detail": f"Voice '{name}' not found."}, status_code=404)
    path.unlink()
    logger.info("Deleted voice '%s'", name)
    return JSONResponse({"deleted": True, "voice_id": name})


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/voices", list_voices, methods=["GET"]),
        Route("/voices/{name}", get_voice, methods=["GET"]),
        Route("/voices", create_voice, methods=["POST"]),
        Route("/voices/{name}", delete_voice, methods=["DELETE"]),
        Route("/health", health, methods=["GET"]),
    ],
)

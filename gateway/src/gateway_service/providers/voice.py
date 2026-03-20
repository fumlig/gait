"""Voice provider — manages WAV reference clips on a local directory.

The voices directory is shared with chatterbox via a Docker volume.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

from gateway_service.models import Voice
from gateway_service.providers.protocols import AudioVoices

if TYPE_CHECKING:
    from typing import Self

    import httpx

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_DEFAULT_VOICE = "default"


class VoiceClient(AudioVoices):
    """Manages voice reference clips on a local directory.

    The "default" voice is virtual — it maps to chatterbox's built-in
    reference clip (no WAV file on disk).
    """

    name = "voices"
    env_var = "VOICES_DIR"

    def __init__(self, voices_dir: str | Path) -> None:
        self._voices_dir = Path(voices_dir)

    @classmethod
    def create(cls, env_value: str, http_client: httpx.AsyncClient) -> Self:
        return cls(voices_dir=env_value)

    def _list_voice_names(self) -> list[str]:
        d = self._voices_dir
        disk_voices = sorted(p.stem for p in d.glob("*.wav")) if d.is_dir() else []
        return [_DEFAULT_VOICE, *[n for n in disk_voices if n != _DEFAULT_VOICE]]

    async def list_voices(self) -> list[Voice]:
        return [Voice(voice_id=n, name=n) for n in self._list_voice_names()]

    async def get_voice(self, name: str) -> Voice:
        if name not in self._list_voice_names():
            raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")
        return Voice(voice_id=name, name=name)

    async def create_voice(self, name: str, audio_data: bytes) -> Voice:
        if not name or not _NAME_RE.match(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Voice name must be non-empty and contain only"
                    " alphanumeric characters, hyphens, or underscores."
                ),
            )

        if name == _DEFAULT_VOICE:
            raise HTTPException(
                status_code=400,
                detail="The 'default' voice is built-in and cannot be replaced.",
            )

        dest = self._voices_dir / f"{name}.wav"
        if dest.exists():
            raise HTTPException(status_code=409, detail=f"Voice '{name}' already exists.")

        if len(audio_data) < 44:
            raise HTTPException(status_code=400, detail="File too small to be a valid WAV file.")

        if audio_data[:4] != b"RIFF" or audio_data[8:12] != b"WAVE":
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid WAV file.")

        self._voices_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(audio_data)
        logger.info("Created voice '%s' (%d bytes)", name, len(audio_data))
        return Voice(voice_id=name, name=name)

    async def delete_voice(self, name: str) -> dict:
        if name == _DEFAULT_VOICE:
            raise HTTPException(
                status_code=400,
                detail="The 'default' voice is built-in and cannot be deleted.",
            )
        path = self._voices_dir / f"{name}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")
        path.unlink()
        logger.info("Deleted voice '%s'", name)
        return {"deleted": True, "voice_id": name}

"""WhisperX backend client — calls the whisperx STT /transcribe and /translate endpoints.

Implements :class:`~gateway_service.protocols.AudioTranscriptions` and
:class:`~gateway_service.protocols.AudioTranslations`.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import HTTPException

from gateway_service.backends.base import BaseBackend
from gateway_service.models import RawSegment, TranscriptionResult, WordTimestamp
from gateway_service.protocols import AudioTranscriptions, AudioTranslations

logger = logging.getLogger(__name__)


class WhisperxClient(BaseBackend, AudioTranscriptions, AudioTranslations):
    """Typed HTTP client for the whisperx transcription (STT) backend.

    Calls whisperx's RPC endpoints:
    - POST /transcribe  (multipart form → JSON)
    - POST /translate   (multipart form → JSON)
    """

    name = "whisperx"
    env_var = "WHISPERX_URL"
    default_model_capabilities: ClassVar[list[str]] = ["transcription"]

    async def transcribe(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
        diarize: bool = False,
    ) -> TranscriptionResult:
        """Forward a transcription request as multipart form data."""
        return await self._stt_request(
            endpoint="/transcribe",
            file=file,
            filename=filename,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            diarize=diarize,
        )

    async def translate(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Forward a translation request as multipart form data."""
        return await self._stt_request(
            endpoint="/translate",
            file=file,
            filename=filename,
            model=model,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )

    # -- Shared helper ------------------------------------------------------

    async def _stt_request(
        self,
        *,
        endpoint: str,
        file: bytes,
        filename: str,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        diarize: bool = False,
    ) -> TranscriptionResult:
        """Send a multipart form request to the backend STT service."""
        url = f"{self._base_url}{endpoint}"

        data: dict[str, str] = {
            "model": model,
            "temperature": str(temperature),
            "word_timestamps": str(word_timestamps).lower(),
            "diarize": str(diarize).lower(),
        }
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt

        files_payload = {"file": (filename, file)}

        resp = await self._http_client.post(url, data=data, files=files_payload)

        if resp.status_code != 200:
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        raw = resp.json()
        return self._parse_result(raw)

    @staticmethod
    def _parse_result(raw: dict) -> TranscriptionResult:
        """Parse the backend's raw JSON into a TranscriptionResult."""
        segments = []
        for seg in raw.get("segments", []):
            words = []
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append(
                        WordTimestamp(
                            word=w.get("word", ""),
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            score=w.get("score"),
                        )
                    )
            segments.append(
                RawSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    words=words,
                    speaker=seg.get("speaker"),
                )
            )
        return TranscriptionResult(
            text=raw.get("text", ""),
            language=raw.get("language", ""),
            duration=raw.get("duration", 0.0),
            segments=segments,
        )

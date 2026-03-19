"""Transcription client — calls the whisperx backend's /transcribe and /translate endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException

from gateway_service.models import ModelObject, RawSegment, TranscriptionResult, WordTimestamp

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class TranscriptionClient:
    """Typed HTTP client for the transcription (STT) backend.

    Calls whisperx's RPC endpoints:
    - POST /transcribe  (multipart form → JSON)
    - POST /translate   (multipart form → JSON)
    - GET /models       (model list)
    - GET /health       (health check)
    """

    def __init__(self, *, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

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

    async def list_models(self) -> list[ModelObject]:
        """Fetch GET /models from the backend."""
        url = f"{self._base_url}/models"
        try:
            resp = await self._client.get(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(
                    "Non-200 from transcription backend (%s): HTTP %d", url, resp.status_code
                )
                return []
            data = resp.json()
            return [ModelObject(**m) for m in data.get("data", [])]
        except Exception:
            logger.warning(
                "Failed to fetch models from transcription backend (%s)", url, exc_info=True
            )
            return []

    async def health(self) -> dict:
        """Check GET /health on the backend."""
        url = f"{self._base_url}/health"
        resp = await self._client.get(url, timeout=5.0)
        if resp.status_code == 200:
            return {"status": "healthy"}
        return {"status": "unhealthy", "detail": f"HTTP {resp.status_code}"}

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
    ) -> TranscriptionResult:
        """Send a multipart form request to the backend STT service.

        The backend returns raw JSON (text, language, duration, segments).
        We parse it into a ``TranscriptionResult``.
        """
        url = f"{self._base_url}{endpoint}"

        # Build multipart form fields.
        data: dict[str, str] = {
            "model": model,
            "temperature": str(temperature),
            "word_timestamps": str(word_timestamps).lower(),
        }
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt

        files_payload = {"file": (filename, file)}

        resp = await self._client.post(url, data=data, files=files_payload)

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

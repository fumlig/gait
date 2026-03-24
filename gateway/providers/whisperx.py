"""WhisperX provider client — STT via /transcribe and /translate endpoints."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar

from fastapi import HTTPException

from gateway.models import RawSegment, TranscriptionResult, WordTimestamp
from gateway.providers.base import BaseProvider
from gateway.providers.protocols import AudioTranscriptions, AudioTranslations

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class WhisperxClient(BaseProvider, AudioTranscriptions, AudioTranslations):
    name = "whisperx"
    url_env = "WHISPERX_URL"
    default_url = "http://whisperx:8000"
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
        return await self._stt_request(
            endpoint="/translate",
            file=file,
            filename=filename,
            model=model,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )

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

        return self._parse_result(resp.json())

    async def transcribe_stream(
        self,
        *,
        file: bytes,
        filename: str,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> AsyncIterator[dict]:
        """Stream segments from the whisperx ``/transcribe_stream`` endpoint.

        Yields dicts with ``start``/``end``/``text`` for each segment,
        then a final metadata dict with ``language``/``duration``.
        """
        url = f"{self._base_url}/transcribe_stream"

        data: dict[str, str] = {
            "model": model,
            "temperature": str(temperature),
        }
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt

        files_payload = {"file": (filename, file)}

        req = self._http_client.build_request(
            "POST", url, data=data, files=files_payload,
        )
        resp = await self._http_client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aread()
            detail = resp.text or f"Backend returned HTTP {resp.status_code}"
            raise HTTPException(status_code=resp.status_code, detail=detail)

        try:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])
        finally:
            await resp.aclose()

    @staticmethod
    def _parse_result(raw: dict) -> TranscriptionResult:
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

"""WhisperX engine: transcription, alignment, and diarization."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

import torch

from whisperx_service.config import settings

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZES = frozenset(
    {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
        "turbo",
    }
)


class WhisperXEngine:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None
        self._align_models: dict[str, tuple[Any, Any]] = {}
        self._diarize_pipeline: Any = None
        self._last_used: float = 0.0

    def load(self, model_name: str | None = None) -> None:
        import whisperx

        name = model_name or settings.default_model
        if not name:
            logger.info("No model name specified, skipping load.")
            return

        logger.info(
            "Loading WhisperX model=%s device=%s compute_type=%s ...",
            name, settings.device, settings.compute_type,
        )
        self._model = whisperx.load_model(
            name, device=settings.device, compute_type=settings.compute_type,
        )
        self._model_name = name
        self.touch()
        logger.info("WhisperX model loaded: %s", name)

    def unload(self) -> None:
        self._model = None
        self._model_name = None
        self._align_models.clear()
        self._diarize_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded.")

    def ensure_model(self, model_name: str) -> None:
        """Ensure the requested model is loaded, swapping if necessary."""
        resolved = self._resolve_model_name(model_name)
        if self._model_name == resolved and self._model is not None:
            return
        logger.info("Swapping model: %s -> %s", self._model_name, resolved)
        self._model = None
        self._model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.load(resolved)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_name(self) -> str | None:
        return self._model_name

    def touch(self) -> None:
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        if not self.is_loaded or self._last_used == 0.0:
            return 0.0
        return time.monotonic() - self._last_used

    def unload_if_idle(self, timeout: float) -> bool:
        if timeout <= 0 or not self.is_loaded:
            return False
        if self.idle_seconds() >= timeout:
            logger.info(
                "Idle for %.0fs (timeout=%ds), unloading models.",
                self.idle_seconds(), timeout,
            )
            self.unload()
            return True
        return False

    def _resolve_model_name(self, name: str) -> str:
        if name == "whisper-1":
            return settings.default_model or "large-v3"
        if name in WHISPER_MODEL_SIZES:
            return name
        return name

    def list_available_models(self) -> list[str]:
        models = ["whisper-1"]
        for size in sorted(WHISPER_MODEL_SIZES):
            if size not in models:
                models.append(size)
        return models

    def _ensure_align_model(self, language_code: str) -> tuple[Any, Any]:
        if language_code in self._align_models:
            return self._align_models[language_code]

        import whisperx

        logger.info("Loading alignment model for language=%s ...", language_code)
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=settings.device,
        )
        self._align_models[language_code] = (model_a, metadata)
        logger.info("Alignment model loaded for language=%s", language_code)
        return model_a, metadata

    def _ensure_diarize_pipeline(self) -> Any:
        if self._diarize_pipeline is not None:
            return self._diarize_pipeline

        from whisperx.diarize import DiarizationPipeline

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for speaker diarization.")

        logger.info("Loading diarization pipeline ...")
        self._diarize_pipeline = DiarizationPipeline(
            token=hf_token, device=settings.device,
        )
        logger.info("Diarization pipeline loaded.")
        return self._diarize_pipeline

    def transcribe(
        self,
        audio_data: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        task: str = "transcribe",
        word_timestamps: bool = False,
        diarize: bool = False,
    ) -> dict[str, Any]:
        """Run the full WhisperX pipeline.

        Returns dict with keys: segments, language, text, duration.
        """
        import whisperx

        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        self.touch()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            audio = whisperx.load_audio(tmp_path)
        finally:
            os.unlink(tmp_path)

        # 1. Transcribe
        transcribe_kwargs: dict[str, Any] = {
            "batch_size": settings.batch_size,
            "task": task,
        }
        if language:
            transcribe_kwargs["language"] = language
        if prompt:
            transcribe_kwargs["prompt"] = prompt
        if temperature > 0:
            transcribe_kwargs["temperature"] = temperature

        logger.info("Transcribing: task=%s, language=%s", task, language)
        result = self._model.transcribe(audio, **transcribe_kwargs)
        detected_language = result.get("language", language or "en")

        # 2. Align (word-level timestamps)
        if word_timestamps:
            try:
                model_a, metadata = self._ensure_align_model(detected_language)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio,
                    settings.device, return_char_alignments=False,
                )
            except Exception:
                logger.warning(
                    "Alignment failed for language=%s, returning without word timestamps",
                    detected_language, exc_info=True,
                )

        # 3. Diarize (speaker labels)
        if diarize and settings.enable_diarization:
            try:
                pipeline = self._ensure_diarize_pipeline()
                diarize_segments = pipeline(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception:
                logger.warning(
                    "Diarization failed, returning without speaker labels",
                    exc_info=True,
                )

        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)
        duration = max((seg.get("end", 0.0) for seg in segments), default=0.0)

        return {
            "segments": segments,
            "language": detected_language,
            "text": full_text,
            "duration": duration,
        }


engine = WhisperXEngine()

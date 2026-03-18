"""WhisperX model loading, inference, alignment, and diarization."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch

from whisperx_service.config import settings
from whisperx_service.models import WHISPER_MODEL_SIZES

logger = logging.getLogger(__name__)


class WhisperXEngine:
    """Manages the WhisperX pipeline: transcription, alignment, diarization."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None
        self._align_models: dict[str, tuple[Any, Any]] = {}  # lang -> (model, metadata)
        self._diarize_pipeline: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_name: str | None = None) -> None:
        """Load a whisper model onto the configured device."""
        import whisperx

        name = model_name or settings.default_model
        logger.info(
            "Loading WhisperX model=%s device=%s compute_type=%s ...",
            name,
            settings.device,
            settings.compute_type,
        )
        self._model = whisperx.load_model(
            name,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        self._model_name = name
        logger.info("WhisperX model loaded: %s", name)

    def unload(self) -> None:
        """Release all model resources."""
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
        # Only unload the whisper model, keep alignment/diarization cached
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

    # ------------------------------------------------------------------
    # Model name resolution
    # ------------------------------------------------------------------

    def _resolve_model_name(self, name: str) -> str:
        """Map OpenAI-compatible aliases to actual whisper model names."""
        if name == "whisper-1":
            return settings.default_model
        if name in WHISPER_MODEL_SIZES:
            return name
        # Allow pass-through for custom model paths/names
        return name

    def list_available_models(self) -> list[str]:
        """Return model IDs that this service reports as available."""
        models = ["whisper-1"]
        if self._model_name and self._model_name not in models:
            models.append(self._model_name)
        return models

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _ensure_align_model(self, language_code: str) -> tuple[Any, Any]:
        """Load and cache the alignment model for a language."""
        if language_code in self._align_models:
            return self._align_models[language_code]

        import whisperx

        logger.info("Loading alignment model for language=%s ...", language_code)
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=settings.device,
        )
        self._align_models[language_code] = (model_a, metadata)
        logger.info("Alignment model loaded for language=%s", language_code)
        return model_a, metadata

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    def _ensure_diarize_pipeline(self) -> Any:
        """Load the pyannote diarization pipeline (requires HF_TOKEN)."""
        if self._diarize_pipeline is not None:
            return self._diarize_pipeline

        from whisperx.diarize import DiarizationPipeline

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for speaker diarization. "
                "Set it to a HuggingFace access token with read permissions for "
                "pyannote/speaker-diarization-community-1."
            )

        logger.info("Loading diarization pipeline ...")
        self._diarize_pipeline = DiarizationPipeline(
            token=hf_token,
            device=settings.device,
        )
        logger.info("Diarization pipeline loaded.")
        return self._diarize_pipeline

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
        """Run the full WhisperX pipeline on audio data.

        Returns a dict with keys: segments, language, text, duration.
        Segments may include word-level timestamps and speaker labels.
        """
        import whisperx

        if self._model is None:
            raise RuntimeError("Model not loaded -- call load() first")

        # Write audio bytes to a temporary file for whisperx
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

        # 2. Align (for word-level timestamps)
        if word_timestamps:
            try:
                model_a, metadata = self._ensure_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    settings.device,
                    return_char_alignments=False,
                )
            except Exception:
                logger.warning(
                    "Alignment failed for language=%s, returning without word timestamps",
                    detected_language,
                    exc_info=True,
                )

        # 3. Diarize (for speaker labels)
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

        # Compute full text from segments
        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)

        # Compute duration from last segment end
        duration = 0.0
        if segments:
            duration = max(seg.get("end", 0.0) for seg in segments)

        return {
            "segments": segments,
            "language": detected_language,
            "text": full_text,
            "duration": duration,
        }


# Module-level singleton
engine = WhisperXEngine()

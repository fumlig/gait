"""WhisperX engine: transcription, alignment, and diarization."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

from whisperx_service.config import settings
from whisperx_service.idle import IdleMixin

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZES = frozenset(
    {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
        "turbo",
    }
)


class WhisperXEngine(IdleMixin):
    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None
        # The most recently loaded model name (persists across unload
        # so we can report ``sleeping`` against the specific model that
        # was auto-unloaded by the idle checker).
        self._last_loaded_name: str | None = None
        self._align_models: dict[str, tuple[Any, Any]] = {}
        self._diarize_pipeline: Any = None

    def load(self, model_name: str | None = None) -> None:
        import whisperx

        name = model_name or settings.default_model
        if not name:
            logger.info("No model name specified, skipping load.")
            return
        resolved = self._resolve_model_name(name)

        logger.info(
            "Loading WhisperX model=%s device=%s compute_type=%s ...",
            resolved, settings.device, settings.compute_type,
        )
        self._last_loaded_name = resolved
        self.mark_loading()
        try:
            self._model = whisperx.load_model(
                resolved,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        except Exception:
            self.mark_unloaded()
            raise
        self._model_name = resolved
        self.mark_loaded()
        self.touch()
        logger.info("WhisperX model loaded: %s", resolved)

    def unload(self) -> None:
        self._model = None
        self._model_name = None
        self._align_models.clear()
        self._diarize_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Manual unload — transition to ``unloaded`` (not ``sleeping``).
        # The idle checker calls ``mark_sleeping`` instead after this.
        self.mark_unloaded()
        logger.info("All models unloaded.")

    def ensure_model(self, model_name: str) -> str:
        """Ensure the requested model is loaded, swapping if necessary.

        Returns the resolved (canonical) model name.
        """
        resolved = self._resolve_model_name(model_name)
        if self._model_name == resolved and self._model is not None:
            return resolved
        logger.info("Swapping model: %s -> %s", self._model_name, resolved)
        self._model = None
        self._model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.load(resolved)
        return resolved

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_name(self) -> str | None:
        return self._model_name

    def status_for(self, model_name: str) -> str:
        """Return the status phase for a specific model id.

        Only one model is resident at a time. The currently-loaded
        model reports ``loaded`` (or ``loading`` mid-load); the most
        recently loaded model reports ``sleeping`` when the engine
        has been auto-unloaded by the idle checker; everything else
        reports ``unloaded``.  ``whisper-1`` is an alias that tracks
        whichever concrete model is currently loaded.
        """
        resolved = self._resolve_model_name(model_name)
        phase = self.status_phase
        if phase == "loading" and resolved == self._last_loaded_name:
            return "loading"
        if resolved == self._model_name and self._model is not None:
            return "loaded"
        if phase == "sleeping" and resolved == self._last_loaded_name:
            return "sleeping"
        return "unloaded"

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

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio_data: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        task: str = "transcribe",
    ) -> Iterator[dict[str, Any]]:
        """Yield segment dicts as the WhisperX pipeline produces them.

        Each yielded dict has ``start``, ``end``, and ``text``.
        The final yield is a metadata dict with ``language`` and
        ``duration`` (no ``text`` key).

        Alignment and diarization are skipped because they require
        all segments up-front and are incompatible with streaming.
        """
        import whisperx
        from whisperx.audio import SAMPLE_RATE
        from whisperx.vads import Pyannote, Vad

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

        pipe = self._model  # FasterWhisperPipeline

        # --- VAD ---
        if issubclass(type(pipe.vad_model), Vad):
            waveform = pipe.vad_model.preprocess_audio(audio)
            merge_fn = pipe.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_fn = Pyannote.merge_chunks

        vad_segments = pipe.vad_model(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        )
        vad_segments = merge_fn(
            vad_segments,
            30,
            onset=pipe._vad_params["vad_onset"],
            offset=pipe._vad_params["vad_offset"],
        )

        if not vad_segments:
            yield {"language": language or "en", "duration": 0.0}
            return

        # --- Tokenizer ---
        from faster_whisper.tokenizer import Tokenizer

        actual_task = task or "transcribe"
        if pipe.tokenizer is None:
            detected_lang = language or pipe.detect_language(audio)
            pipe.tokenizer = Tokenizer(
                pipe.model.hf_tokenizer,
                pipe.model.model.is_multilingual,
                task=actual_task,
                language=detected_lang,
            )
        else:
            detected_lang = language or pipe.tokenizer.language_code
            if (
                actual_task != pipe.tokenizer.task
                or detected_lang != pipe.tokenizer.language_code
            ):
                pipe.tokenizer = Tokenizer(
                    pipe.model.hf_tokenizer,
                    pipe.model.model.is_multilingual,
                    task=actual_task,
                    language=detected_lang,
                )

        # --- Prompt ---
        saved_options = None
        if prompt:
            from dataclasses import replace

            saved_options = pipe.options
            pipe.options = replace(pipe.options, initial_prompt=prompt)

        # --- Audio chunk generator ---
        def audio_chunks():
            for seg in vad_segments:
                f1 = int(seg["start"] * SAMPLE_RATE)
                f2 = int(seg["end"] * SAMPLE_RATE)
                yield {"inputs": audio[f1:f2]}

        # --- Iterate pipeline (yields per-segment results) ---
        batch_size = settings.batch_size
        duration = 0.0
        try:
            logger.info(
                "Streaming transcription: task=%s, language=%s, segments=%d",
                actual_task, detected_lang, len(vad_segments),
            )
            for idx, out in enumerate(
                pipe(audio_chunks(), batch_size=batch_size, num_workers=0),
            ):
                text = out["text"]
                if batch_size in (0, 1, None):
                    text = text[0]
                end = round(vad_segments[idx]["end"], 3)
                if end > duration:
                    duration = end
                yield {
                    "start": round(vad_segments[idx]["start"], 3),
                    "end": end,
                    "text": text.strip() if isinstance(text, str) else str(text),
                }
        finally:
            if saved_options is not None:
                pipe.options = saved_options
            if pipe.preset_language is None:
                pipe.tokenizer = None

        yield {"language": detected_lang, "duration": duration}


engine = WhisperXEngine()

"""Chatterbox-Turbo model loading and inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torchaudio

from chatterbox_service.config import settings

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ChatterboxEngine:
    """Wraps ChatterboxTurboTTS for use by the API layer."""

    def __init__(self) -> None:
        self._model = None
        self._sample_rate: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Chatterbox-Turbo model onto the configured device."""
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        logger.info("Loading Chatterbox-Turbo on device=%s ...", settings.device)
        self._model = ChatterboxTurboTTS.from_pretrained(device=settings.device)
        self._sample_rate = self._model.sr
        logger.info("Model loaded. Sample rate=%d", self._sample_rate)

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("Model not loaded")
        return self._sample_rate

    # ------------------------------------------------------------------
    # Voice resolution
    # ------------------------------------------------------------------

    def list_voices(self) -> list[str]:
        """Return the names of all registered voices (stem of .wav files)."""
        voices_dir = settings.voices_dir
        if not voices_dir.is_dir():
            return []
        return sorted(p.stem for p in voices_dir.glob("*.wav"))

    def _resolve_voice(self, voice: str) -> Path | None:
        """Map a voice name to its reference .wav path, or None for default."""
        path = settings.voices_dir / f"{voice}.wav"
        if path.is_file():
            return path
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, text: str, voice: str, speed: float = 1.0) -> tuple[torch.Tensor, int]:
        """Generate speech from text.

        Returns:
            (waveform_tensor, sample_rate)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        ref_path = self._resolve_voice(voice)
        kwargs: dict = {"text": text}
        if ref_path is not None:
            kwargs["audio_prompt_path"] = str(ref_path)

        logger.info("Generating speech: voice=%s, speed=%.2f, len=%d", voice, speed, len(text))
        wav = self._model.generate(**kwargs)

        # Apply speed adjustment via resampling if speed != 1.0
        sr = self.sample_rate
        if speed != 1.0:
            wav, sr = _apply_speed(wav, sr, speed)

        return wav, sr


def _apply_speed(wav: torch.Tensor, sr: int, speed: float) -> tuple[torch.Tensor, int]:
    """Change playback speed by resampling.

    We resample *up* by ``speed`` then declare the original sample rate,
    which effectively speeds up / slows down playback without pitch shift
    beyond what resampling introduces.
    """
    new_sr = int(sr * speed)
    resampler = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sr)
    wav = resampler(wav)
    return wav, sr


# Module-level singleton — initialised during app lifespan.
engine = ChatterboxEngine()

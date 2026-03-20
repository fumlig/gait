"""Multi-model Chatterbox engine: Turbo, Original, and Multilingual."""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchaudio

from chatterbox_service.config import settings

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# GPU optimisations (safe on all devices, only meaningful on CUDA Ampere+)
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # use TF32 for matmuls
    torch.backends.cudnn.benchmark = True  # auto-tune cudnn kernels

# ---------------------------------------------------------------------------
# Language support
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

KNOWN_MODELS = frozenset(
    {
        "chatterbox-turbo",
        "chatterbox",
        "chatterbox-multilingual",
    }
)

# Aliases accepted for OpenAI drop-in compatibility.
MODEL_ALIASES: dict[str, str] = {
    "tts-1": "chatterbox-turbo",
    "tts-1-hd": "chatterbox",
}

# Models that support the exaggeration / cfg_weight controls.
_CFG_MODELS = frozenset({"chatterbox", "chatterbox-multilingual"})

# Models that require a language parameter.
_MULTILINGUAL_MODELS = frozenset({"chatterbox-multilingual"})


def resolve_model_name(name: str) -> str:
    """Map OpenAI-compatible aliases to the canonical model name."""
    return MODEL_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ChatterboxEngine:
    """Manages multiple Chatterbox model variants.

    Models are loaded lazily on first request.  The ``default_model`` is
    optionally preloaded during Starlette lifespan startup.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._sample_rates: dict[str, int] = {}
        self._last_used: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_name: str | None = None) -> None:
        """Load a specific model by canonical name.

        If *model_name* is ``None``, loads ``settings.default_model``.
        """
        model_name = model_name or settings.default_model
        model_name = resolve_model_name(model_name)

        if model_name in self._models:
            logger.info("Model %s already loaded, skipping.", model_name)
            return

        logger.info("Loading %s on device=%s ...", model_name, settings.device)

        if model_name == "chatterbox-turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            model = ChatterboxTurboTTS.from_pretrained(device=settings.device)

        elif model_name == "chatterbox":
            from chatterbox.tts import ChatterboxTTS

            model = ChatterboxTTS.from_pretrained(device=settings.device)

        elif model_name == "chatterbox-multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            model = ChatterboxMultilingualTTS.from_pretrained(device=settings.device)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self._models[model_name] = model
        self._sample_rates[model_name] = model.sr
        self.touch()
        logger.info("Model %s loaded. Sample rate=%d", model_name, model.sr)

    def unload(self, model_name: str | None = None) -> None:
        """Release one or all model(s)."""
        names = [model_name] if model_name else list(self._models.keys())
        for name in names:
            if name in self._models:
                del self._models[name]
                self._sample_rates.pop(name, None)
                logger.info("Model %s unloaded.", name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_model(self, model_name: str) -> str:
        """Validate, resolve aliases, and load if needed.

        Returns the canonical model name.

        Raises ``ValueError`` for unknown model names.
        """
        resolved = resolve_model_name(model_name)
        if resolved not in KNOWN_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {', '.join(sorted(KNOWN_MODELS))}"
            )
        if resolved not in self._models:
            self.load(resolved)
        return resolved

    # ------------------------------------------------------------------
    # Idle tracking
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Record that the engine was just used (reset idle timer)."""
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        """Return the number of seconds since last use, or 0 if never used."""
        if not self.is_loaded or self._last_used == 0.0:
            return 0.0
        return time.monotonic() - self._last_used

    def unload_if_idle(self, timeout: float) -> bool:
        """Unload all models if idle longer than *timeout* seconds.

        Returns True if models were unloaded.
        """
        if timeout <= 0 or not self.is_loaded:
            return False
        if self.idle_seconds() >= timeout:
            logger.info(
                "Idle for %.0fs (timeout=%ds), unloading models.",
                self.idle_seconds(),
                timeout,
            )
            self.unload()
            return True
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def loaded_models(self) -> dict[str, bool]:
        """Return a dict of model_name -> loaded status for all known models."""
        return {name: name in self._models for name in sorted(KNOWN_MODELS)}

    @property
    def is_loaded(self) -> bool:
        """True if at least one model is loaded."""
        return bool(self._models)

    def sample_rate(self, model_name: str) -> int:
        sr = self._sample_rates.get(model_name)
        if sr is None:
            raise RuntimeError(f"Model {model_name} not loaded")
        return sr

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
        """Map a voice name to its reference .wav path, or None for default.

        The special name ``"default"`` always resolves to ``None``, which
        tells the chatterbox library to use its built-in default voice.
        """
        if voice == "default":
            return None
        path = settings.voices_dir / f"{voice}.wav"
        if path.is_file():
            return path
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        model_name: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        top_p: float = 1.0,
        min_p: float = 0.05,
        top_k: int = 1000,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Generate speech from text using the specified model.

        Returns (waveform_tensor, sample_rate).
        """
        model = self._models.get(model_name)
        if model is None:
            raise RuntimeError(f"Model {model_name} not loaded — call load() first")

        self.touch()

        # Set seed for reproducibility
        if seed is not None and seed != 0:
            _set_seed(seed)

        ref_path = self._resolve_voice(voice)
        audio_prompt_path = str(ref_path) if ref_path is not None else None

        sr = self._sample_rates[model_name]

        with torch.inference_mode():
            if model_name == "chatterbox-turbo":
                wav = self._generate_turbo(
                    model,
                    text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                )

            elif model_name == "chatterbox":
                wav = self._generate_original(
                    model,
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    min_p=min_p,
                )

            elif model_name == "chatterbox-multilingual":
                wav = self._generate_multilingual(
                    model,
                    text,
                    language=language or "en",
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    min_p=min_p,
                )
            else:
                raise RuntimeError(f"No generate implementation for {model_name}")

        # Apply speed adjustment via resampling if speed != 1.0
        if speed != 1.0:
            wav, sr = _apply_speed(wav, sr, speed)

        return wav, sr

    # ------------------------------------------------------------------
    # Per-model generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_turbo(
        model: Any,
        text: str,
        *,
        audio_prompt_path: str | None = None,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        top_p: float = 0.95,
        top_k: int = 1000,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "text": text,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
        }
        if audio_prompt_path is not None:
            kwargs["audio_prompt_path"] = audio_prompt_path
        return model.generate(**kwargs)

    @staticmethod
    def _generate_original(
        model: Any,
        text: str,
        *,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        top_p: float = 1.0,
        min_p: float = 0.05,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "min_p": min_p,
        }
        if audio_prompt_path is not None:
            kwargs["audio_prompt_path"] = audio_prompt_path
        return model.generate(**kwargs)

    @staticmethod
    def _generate_multilingual(
        model: Any,
        text: str,
        *,
        language: str = "en",
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        top_p: float = 1.0,
        min_p: float = 0.05,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "text": text,
            "language_id": language,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "min_p": min_p,
        }
        if audio_prompt_path is not None:
            kwargs["audio_prompt_path"] = audio_prompt_path
        return model.generate(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def validate_language(model_name: str, language: str | None) -> None:
    """Raise ``ValueError`` if language config is invalid for the model."""
    if model_name in _MULTILINGUAL_MODELS:
        if not language:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(f"'language' is required for {model_name}. Supported: {supported}")
        if language.lower() not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(f"Unsupported language '{language}'. Supported: {supported}")


# Module-level singleton — initialised during app lifespan.
engine = ChatterboxEngine()

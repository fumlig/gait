"""Text preprocessing for TTS input.

Strips symbols that degrade speech synthesis quality while preserving
normal prose characters and [paralinguistic tags] like [laugh], [cough].
"""

from __future__ import annotations

import re

# Characters allowed through: word chars, whitespace, common punctuation,
# and square brackets (for paralinguistic tags).
_ALLOWED = re.compile(r"[^\w\s.,;:!?'\"()\-\[\]\u2026\u2013\u2014/]", re.UNICODE)

# Collapse runs of whitespace (except single newlines) into one space.
_EXTRA_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{2,}")


def preprocess_speech_text(text: str) -> str:
    """Clean *text* for TTS, keeping prose and [paralinguistic tags]."""
    text = _ALLOWED.sub("", text)
    text = _EXTRA_WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n", text)
    return text.strip()

"""POST /v1/chat/completions — proxied to the chat provider.

When the request includes ``input_audio`` content parts the gateway runs
STT to obtain text before forwarding to the LLM.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from gateway.deps import (
    ChatClient,
    backend_errors,
    require_transcription,
)
from gateway.formatting import pcm16_to_wav
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

    from gateway.providers.protocols import AudioTranscriptions

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers — model auto-detection
# ---------------------------------------------------------------------------


def _find_stt_model(request: Request) -> str:
    """Auto-detect an STT model from the cached model list."""
    models = getattr(request.app.state, "models", [])
    for model in models:
        caps = getattr(model, "capabilities", []) or []
        mid = model.id.lower()
        if "transcription" in caps or "whisper" in mid or "stt" in mid:
            return model.id
    raise HTTPException(
        status_code=400,
        detail="No STT model available to transcribe input audio.",
    )


# ---------------------------------------------------------------------------
# Helpers — input_audio preprocessing
# ---------------------------------------------------------------------------


def _has_input_audio(messages: list[ChatMessage]) -> bool:
    """Return True if any message contains an ``input_audio`` content part."""
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") == "input_audio":
                return True
    return False


async def _preprocess_input_audio(
    messages: list[ChatMessage],
    transcription_client: AudioTranscriptions,
    stt_model: str,
) -> list[ChatMessage]:
    """Replace ``input_audio`` content parts with transcribed text.

    Returns a new list — the originals are not mutated.
    """
    result: list[ChatMessage] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            result.append(msg)
            continue

        new_parts: list[dict] = []
        changed = False
        for part in msg.content:
            if not isinstance(part, dict) or part.get("type") != "input_audio":
                new_parts.append(part)
                continue

            changed = True
            audio_info = part.get("input_audio") or {}
            b64_data: str = audio_info.get("data", "")
            fmt: str = audio_info.get("format", "wav")

            if not b64_data:
                raise HTTPException(
                    status_code=400,
                    detail="input_audio content part has empty 'data' field.",
                )

            try:
                raw_bytes = base64.b64decode(b64_data)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="input_audio 'data' is not valid base64.",
                ) from exc

            # Convert raw PCM16 → WAV so the STT backend can process it.
            wav_bytes = pcm16_to_wav(raw_bytes) if fmt == "pcm16" else raw_bytes

            async with backend_errors("Input audio transcription"):
                tr = await transcription_client.transcribe(
                    file=wav_bytes,
                    filename="input_audio.wav",
                    model=stt_model,
                    language=None,
                    prompt=None,
                    temperature=0.0,
                    word_timestamps=False,
                )

            new_parts.append({"type": "text", "text": tr.text})

        if changed:
            result.append(msg.model_copy(update={"content": new_parts}))
        else:
            result.append(msg)

    return result


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_unset=True,
)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    client: ChatClient,
) -> ChatCompletionResponse | StreamingResponse:
    # --- input_audio preprocessing -------------------------------------------
    if _has_input_audio(body.messages):
        stt_client: AudioTranscriptions = require_transcription(request)
        stt_model = _find_stt_model(request)
        body = body.model_copy(
            update={
                "messages": await _preprocess_input_audio(
                    body.messages, stt_client, stt_model,
                ),
            },
        )

    # --- dispatch ------------------------------------------------------------
    async with backend_errors("Chat completion"):
        if body.stream:
            return await client.chat_completions_stream(body)
        return await client.chat_completions(body)

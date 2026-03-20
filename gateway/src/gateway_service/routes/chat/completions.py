"""POST /v1/chat/completions — proxied to llama.cpp server.

When the request includes modalities: ["text", "audio"], the gateway streams
text from the LLM and interleaves synthesised PCM16 audio from the speech
backend, matching the OpenAI streaming audio contract.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from gateway_service.formatting import wav_to_pcm16
from gateway_service.models import SpeechRequest, SpeechResponseFormat

if TYPE_CHECKING:
    from gateway_service.providers.protocols import AudioSpeech, ChatCompletions

logger = logging.getLogger(__name__)

router = APIRouter()

# Sentence boundary: punctuation followed by whitespace, or newline.
_BOUNDARY = re.compile(r"[.!?]\s|\n")
_MAX_SENTENCE_BUF = 500
_AUDIO_CHUNK_BYTES = 8192  # ~170 ms at 24 kHz mono PCM16


def _extract_sentences(buf: str) -> tuple[str, str]:
    """Split buf at the last sentence boundary into (complete, remainder)."""
    matches = list(_BOUNDARY.finditer(buf))
    if matches:
        last = matches[-1]
        return buf[: last.end()].strip(), buf[last.end() :]

    if len(buf) > _MAX_SENTENCE_BUF:
        space = buf.rfind(" ", 0, _MAX_SENTENCE_BUF)
        if space > 0:
            return buf[:space].strip(), buf[space + 1 :]
        return buf[:_MAX_SENTENCE_BUF].strip(), buf[_MAX_SENTENCE_BUF:]

    return "", buf


def _get_chat_client(request: Request) -> ChatCompletions:
    client = getattr(request.app.state, "chat_completions", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No chat backend configured.")
    return client


def _get_speech_client(request: Request) -> AudioSpeech:
    client = getattr(request.app.state, "audio_speech", None)
    if client is None:
        raise HTTPException(status_code=503, detail="No speech backend configured.")
    return client


def _find_tts_model(request: Request) -> str:
    """Auto-detect a TTS model from the cached model list."""
    models = getattr(request.app.state, "models", [])
    for model in models:
        mid = model.id.lower()
        if "chatterbox" in mid or "tts" in mid:
            return model.id
    raise HTTPException(
        status_code=400,
        detail="No TTS model available for audio output. Specify 'audio.model' in the request.",
    )


def _audio_sse(template: dict, **audio_fields: object) -> str:
    event = {
        **template,
        "choices": [
            {"index": 0, "delta": {"audio": audio_fields}, "finish_reason": None}
        ],
    }
    return f"data: {json.dumps(event)}\n\n"


async def _synth_and_emit(
    speech_client: AudioSpeech,
    text: str,
    tts_model: str,
    voice: str,
    template: dict,
    audio_id: str,
):
    """Synthesise text, yield audio-data and transcript SSE events.

    Errors are logged but swallowed so text streaming continues.
    """
    try:
        req = SpeechRequest(
            model=tts_model, input=text, voice=voice,
            response_format=SpeechResponseFormat.wav,
        )
        wav_bytes, _ = await speech_client.synthesize(req)
        pcm_data, _sr = wav_to_pcm16(wav_bytes)

        for i in range(0, len(pcm_data), _AUDIO_CHUNK_BYTES):
            chunk = pcm_data[i : i + _AUDIO_CHUNK_BYTES]
            yield _audio_sse(template, data=base64.b64encode(chunk).decode())

        yield _audio_sse(template, transcript=text)
    except Exception:
        logger.exception("Audio synthesis failed for chunk: %s", text[:80])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    client = _get_chat_client(request)
    body = await request.json()

    wants_audio = "audio" in (body.get("modalities") or [])

    if wants_audio:
        if not body.get("stream"):
            raise HTTPException(
                status_code=400,
                detail="Audio modality currently requires stream=true.",
            )
        return await _chat_with_audio(request, client, body)

    try:
        if body.get("stream"):
            return await client.chat_completions_stream(body)
        result = await client.chat_completions(body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=502, detail="Chat backend unavailable.") from exc


async def _chat_with_audio(
    request: Request, chat_client: ChatCompletions, body: dict,
) -> StreamingResponse:
    """Stream text from the LLM and interleave TTS audio."""
    speech_client = _get_speech_client(request)

    audio_cfg = body.get("audio", {})
    voice = audio_cfg.get("voice", "default")
    tts_model = audio_cfg.get("model") or _find_tts_model(request)

    # Strip audio-specific fields before forwarding to the LLM
    llm_body = {k: v for k, v in body.items() if k not in ("modalities", "audio")}

    try:
        llm_resp = await chat_client.chat_completions_stream_raw(llm_body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat completion failed (audio path)")
        raise HTTPException(status_code=502, detail="Chat backend unavailable.") from exc

    async def _generate():
        try:
            audio_id = uuid4().hex[:16]
            template: dict | None = None
            sentence_buf = ""

            async for line in llm_resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                payload = line[6:]
                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if template is None:
                    template = {
                        "id": chunk.get("id", ""),
                        "object": chunk.get("object", "chat.completion.chunk"),
                        "created": chunk.get("created", int(time.time())),
                        "model": chunk.get("model", ""),
                    }
                    yield _audio_sse(template, id=audio_id)

                yield f"data: {json.dumps(chunk)}\n\n"

                content = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if content:
                    sentence_buf += content
                    complete, sentence_buf = _extract_sentences(sentence_buf)
                    if complete:
                        async for evt in _synth_and_emit(
                            speech_client, complete, tts_model, voice, template, audio_id,
                        ):
                            yield evt

            # Flush remaining text
            if sentence_buf.strip() and template:
                async for evt in _synth_and_emit(
                    speech_client, sentence_buf.strip(), tts_model, voice, template, audio_id,
                ):
                    yield evt

            if template:
                yield _audio_sse(template, expires_at=int(time.time()) + 3600)

            yield "data: [DONE]\n\n"
        finally:
            await llm_resp.aclose()

    return StreamingResponse(_generate(), media_type="text/event-stream")

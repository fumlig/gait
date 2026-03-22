"""POST /v1/chat/completions — proxied to llama.cpp server.

When the request includes ``input_audio`` content parts the gateway runs
STT to obtain text before forwarding to the LLM.

When the request includes ``modalities: ["text", "audio"]`` the gateway
synthesises speech from the LLM's text output, matching the OpenAI audio
contract for both streaming and non-streaming responses.
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
from starlette.responses import StreamingResponse

from gateway.deps import (
    ChatClient,
    backend_errors,
    require_speech,
    require_transcription,
)
from gateway.formatting import pcm16_to_wav, wav_to_pcm16
from gateway.models import (
    ChatAudioConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageAudio,
    SpeechRequest,
    SpeechResponseFormat,
)

if TYPE_CHECKING:
    from gateway.providers.protocols import AudioSpeech, AudioTranscriptions, ChatCompletions

logger = logging.getLogger(__name__)

router = APIRouter()

# Sentence boundary: punctuation followed by whitespace, or newline.
_BOUNDARY = re.compile(r"[.!?]\s|\n")
_MAX_SENTENCE_BUF = 500
_AUDIO_CHUNK_BYTES = 8192  # ~170 ms at 24 kHz mono PCM16


# ---------------------------------------------------------------------------
# Helpers — sentence extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers — model auto-detection
# ---------------------------------------------------------------------------


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
# Helpers — audio SSE formatting
# ---------------------------------------------------------------------------


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
    wants_audio = body.modalities is not None and "audio" in body.modalities

    if wants_audio:
        if body.stream:
            return await _chat_with_audio_streaming(request, client, body)
        return await _chat_with_audio_nonstreaming(request, client, body)

    async with backend_errors("Chat completion"):
        if body.stream:
            return await client.chat_completions_stream(body)
        return await client.chat_completions(body)


# ---------------------------------------------------------------------------
# Audio output — non-streaming
# ---------------------------------------------------------------------------


async def _chat_with_audio_nonstreaming(
    request: Request,
    chat_client: ChatCompletions,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Get full LLM response, synthesise speech, return message.audio."""
    speech_client: AudioSpeech = require_speech(request)

    audio_cfg = body.audio or ChatAudioConfig()
    voice = audio_cfg.voice
    tts_model = audio_cfg.model or _find_tts_model(request)

    # Build LLM request without audio-specific fields
    llm_body = ChatCompletionRequest.model_validate(
        body.model_dump(exclude_unset=True, exclude={"modalities", "audio"}),
    )

    async with backend_errors("Chat completion"):
        resp = await chat_client.chat_completions(llm_body)

    # Extract assistant text
    raw_content = resp.choices[0].message.content if resp.choices else None
    text = raw_content if isinstance(raw_content, str) else ""

    # Synthesise audio from the full text
    audio_id = uuid4().hex[:16]
    try:
        req = SpeechRequest(
            model=tts_model, input=text, voice=voice,
            response_format=SpeechResponseFormat.wav,
        )
        wav_bytes, _ = await speech_client.synthesize(req)
        pcm_data, _sr = wav_to_pcm16(wav_bytes)
        b64_audio = base64.b64encode(pcm_data).decode()
    except Exception as exc:
        logger.exception("Audio synthesis failed for non-streaming response")
        raise HTTPException(
            status_code=502,
            detail="Audio synthesis failed.",
        ) from exc

    # Build the audio attachment and clear content per OpenAI contract
    audio_obj = ChatMessageAudio(
        id=audio_id,
        data=b64_audio,
        transcript=text,
        expires_at=int(time.time()) + 3600,
    )
    if resp.choices:
        msg = resp.choices[0].message.model_copy(
            update={"content": None, "audio": audio_obj},
        )
        choice = resp.choices[0].model_copy(update={"message": msg})
        resp = resp.model_copy(update={"choices": [choice]})

    return resp


# ---------------------------------------------------------------------------
# Audio output — streaming
# ---------------------------------------------------------------------------


async def _chat_with_audio_streaming(
    request: Request,
    chat_client: ChatCompletions,
    body: ChatCompletionRequest,
) -> StreamingResponse:
    """Stream text from the LLM and interleave TTS audio."""
    speech_client: AudioSpeech = require_speech(request)

    audio_cfg = body.audio or ChatAudioConfig()
    voice = audio_cfg.voice
    tts_model = audio_cfg.model or _find_tts_model(request)

    # Build LLM request without audio-specific fields
    llm_body = ChatCompletionRequest.model_validate(
        body.model_dump(exclude_unset=True, exclude={"modalities", "audio"}),
    )

    async with backend_errors("Chat completion"):
        llm_resp = await chat_client.chat_completions_stream_raw(llm_body)

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

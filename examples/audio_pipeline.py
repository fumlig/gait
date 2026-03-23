#!/usr/bin/env python3
"""Client-driven audio conversation pipeline for gait.

Demonstrates two modes:

  sequential  — transcribe → respond → speak (one step at a time)
  pipelined   — transcribe → stream response + TTS per sentence (overlapped)

The pipelined mode starts producing audio as soon as the first sentence
is generated, while the LLM continues.  Timeline comparison:

  Sequential:
    [===STT===][========LLM========][========TTS========]  → total

  Pipelined:
    [===STT===][========LLM streaming========]
                    [TTS-1]   [TTS-2]  [TTS-3]            → STT + first-sentence + TTS-1

Requirements: pip install httpx (or: uv pip install httpx)

Usage:
    # From text (skip STT)
    python examples/audio_pipeline.py sequential --text "Explain black holes briefly."
    python examples/audio_pipeline.py pipelined  --text "Explain black holes briefly."

    # From audio file
    python examples/audio_pipeline.py sequential recording.wav
    python examples/audio_pipeline.py pipelined  recording.wav

    # Specify models / voice / output
    python examples/audio_pipeline.py pipelined recording.wav \
        --llm-model my-model \
        --stt-model large-v3 \
        --tts-model chatterbox \
        --voice alice \
        --output response.wav

    # With system instructions
    python examples/audio_pipeline.py pipelined --text "Hi there" \
        --instructions "You are a helpful assistant. Keep answers short."
"""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import io
import json
import re
import sys
import time
import wave

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:3000"
TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# Sentence splitting: break on . ! ? followed by whitespace, but only
# after accumulating at least MIN_CHUNK_CHARS to avoid tiny TTS calls.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MIN_CHUNK_CHARS = 40


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


async def discover_models(
    client: httpx.AsyncClient,
    base_url: str,
) -> dict[str, str]:
    """Fetch /v1/models and return {capability: model_id} for the first
    model that advertises each capability we care about."""
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    models = resp.json()["data"]

    found: dict[str, str] = {}
    for m in models:
        for cap in m.get("capabilities", []):
            if cap not in found:
                found[cap] = m["id"]
    return found


async def transcribe(
    client: httpx.AsyncClient,
    base_url: str,
    audio: bytes,
    *,
    model: str,
) -> str:
    """POST /v1/audio/transcriptions → transcribed text."""
    resp = await client.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"model": model, "response_format": "json"},
    )
    resp.raise_for_status()
    return resp.json()["text"]


async def respond(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    *,
    model: str,
    instructions: str | None = None,
) -> str:
    """POST /v1/responses (non-streaming) → response text."""
    body: dict = {"model": model, "input": text}
    if instructions:
        body["instructions"] = instructions

    resp = await client.post(f"{base_url}/v1/responses", json=body)
    resp.raise_for_status()
    data = resp.json()

    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part["text"]
    return ""


async def speak(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    *,
    model: str,
    voice: str,
) -> bytes:
    """POST /v1/audio/speech → WAV bytes."""
    resp = await client.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "wav",
        },
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# SSE stream parsing
# ---------------------------------------------------------------------------


async def iter_text_deltas(
    response: httpx.Response,
) -> collections.abc.AsyncGenerator[str, None]:
    """Yield text delta strings from a Responses API SSE stream.

    Handles the standard OpenAI event format::

        event: response.output_text.delta
        data: {"type":"response.output_text.delta","delta":"Hello"}
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.output_text.delta":
            yield event["delta"]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_at_sentence(buf: str) -> tuple[str | None, str]:
    """Try to split a complete sentence from *buf*.

    Returns ``(sentence, remainder)``.  *sentence* is ``None`` when no
    boundary has been found yet or the buffer is too short.
    """
    # Walk matches from the end so we yield the longest possible chunk.
    for m in reversed(list(_SENTENCE_END.finditer(buf))):
        if m.start() >= _MIN_CHUNK_CHARS:
            return buf[: m.start() + 1].strip(), buf[m.end() :]
    return None, buf


# ---------------------------------------------------------------------------
# WAV concatenation
# ---------------------------------------------------------------------------


def concatenate_wav(chunks: list[bytes]) -> bytes:
    """Join multiple WAV files (same format) into one."""
    all_frames: list[bytes] = []
    params = None
    for chunk in chunks:
        with wave.open(io.BytesIO(chunk), "rb") as wf:
            if params is None:
                params = wf.getparams()
            all_frames.append(wf.readframes(wf.getnframes()))

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        assert params is not None
        wf.setparams(params)
        for frames in all_frames:
            wf.writeframes(frames)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Pipeline: sequential
# ---------------------------------------------------------------------------


async def run_sequential(
    *,
    audio: bytes | None,
    text: str | None,
    base_url: str,
    llm_model: str,
    stt_model: str,
    tts_model: str,
    voice: str,
    instructions: str | None,
    output_path: str,
) -> None:
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # ① Transcribe ---------------------------------------------------
        if audio is not None:
            print("① Transcribing…")
            t = time.monotonic()
            text = await transcribe(client, base_url, audio, model=stt_model)
            print(f"  → \"{text}\"  ({time.monotonic() - t:.1f}s)")

        assert text is not None

        # ② Generate response --------------------------------------------
        print("② Generating response…")
        t = time.monotonic()
        response_text = await respond(
            client, base_url, text,
            model=llm_model, instructions=instructions,
        )
        elapsed = time.monotonic() - t
        preview = response_text[:120] + ("…" if len(response_text) > 120 else "")
        print(f"  → \"{preview}\"  ({elapsed:.1f}s)")

        # ③ Synthesize speech ---------------------------------------------
        print("③ Synthesizing speech…")
        t = time.monotonic()
        wav = await speak(
            client, base_url, response_text,
            model=tts_model, voice=voice,
        )
        print(f"  → {output_path} ({len(wav):,} bytes)  ({time.monotonic() - t:.1f}s)")

    with open(output_path, "wb") as f:
        f.write(wav)

    print(f"\nTotal: {time.monotonic() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Pipeline: pipelined (streaming LLM + concurrent TTS)
# ---------------------------------------------------------------------------


async def run_pipelined(
    *,
    audio: bytes | None,
    text: str | None,
    base_url: str,
    llm_model: str,
    stt_model: str,
    tts_model: str,
    voice: str,
    instructions: str | None,
    output_path: str,
) -> None:
    t0 = time.monotonic()
    audio_chunks: list[bytes] = []

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # ① Transcribe (blocking — need full text before LLM) ------------
        if audio is not None:
            print("① Transcribing…")
            t = time.monotonic()
            text = await transcribe(client, base_url, audio, model=stt_model)
            print(f"  → \"{text}\"  ({time.monotonic() - t:.1f}s)")

        assert text is not None

        # ② + ③  Stream LLM, pipeline TTS at sentence boundaries ---------
        print("②③ Streaming response + pipelined TTS…")
        t_stream = time.monotonic()

        # TTS worker: consumes sentences from a queue, preserves order.
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        chunk_idx = 0

        async def tts_worker() -> None:
            nonlocal chunk_idx
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                chunk_idx += 1
                t = time.monotonic()
                wav = await speak(
                    client, base_url, sentence,
                    model=tts_model, voice=voice,
                )
                audio_chunks.append(wav)
                elapsed = time.monotonic() - t
                preview = sentence[:60] + ("…" if len(sentence) > 60 else "")
                print(f"  🔊 chunk {chunk_idx} ({elapsed:.1f}s): \"{preview}\"")

        worker = asyncio.create_task(tts_worker())

        # Stream the Responses API
        body: dict = {"model": llm_model, "input": text, "stream": True}
        if instructions:
            body["instructions"] = instructions

        buf = ""
        full_text = ""

        async with client.stream(
            "POST", f"{base_url}/v1/responses", json=body,
        ) as resp:
            resp.raise_for_status()
            async for delta in iter_text_deltas(resp):
                buf += delta
                full_text += delta

                sentence, buf = split_at_sentence(buf)
                if sentence:
                    await tts_queue.put(sentence)

        # Flush remaining text
        if buf.strip():
            await tts_queue.put(buf.strip())

        # Signal worker to finish, then wait
        await tts_queue.put(None)
        await worker

        elapsed = time.monotonic() - t_stream
        preview = full_text[:120] + ("…" if len(full_text) > 120 else "")
        print(f"\n  Full response ({elapsed:.1f}s): \"{preview}\"")

    # Concatenate all WAV chunks into one file
    if audio_chunks:
        combined = concatenate_wav(audio_chunks)
        with open(output_path, "wb") as f:
            f.write(combined)
        print(f"  → {output_path} ({len(combined):,} bytes)")

    print(f"\nTotal: {time.monotonic() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Client-driven audio pipeline for gait",
    )
    p.add_argument(
        "mode",
        choices=["sequential", "pipelined"],
        help="Pipeline mode",
    )
    p.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Input audio file (WAV). Omit if using --text.",
    )
    p.add_argument("--text", default=None, help="Input text (skip STT)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Gait gateway URL")
    p.add_argument("--llm-model", default=None, help="LLM model id")
    p.add_argument("--stt-model", default=None, help="STT model id")
    p.add_argument("--tts-model", default=None, help="TTS model id")
    p.add_argument("--voice", default="default", help="TTS voice name")
    p.add_argument("--instructions", default=None, help="System instructions")
    p.add_argument("--output", default="response.wav", help="Output WAV path")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Load audio file if provided
    audio: bytes | None = None
    if args.audio_file:
        with open(args.audio_file, "rb") as f:
            audio = f.read()
        print(f"Loaded {args.audio_file} ({len(audio):,} bytes)")
    elif args.text is None:
        print("Error: provide an audio file or --text", file=sys.stderr)
        sys.exit(1)

    # Auto-detect models from the gateway if not specified
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        models = await discover_models(client, args.base_url)

    llm_model = args.llm_model or models.get("chat")
    stt_model = args.stt_model or models.get("transcription")
    tts_model = args.tts_model or models.get("speech")

    if not llm_model:
        print("Error: no LLM model found. Use --llm-model.", file=sys.stderr)
        sys.exit(1)
    if audio and not stt_model:
        print("Error: no STT model found. Use --stt-model.", file=sys.stderr)
        sys.exit(1)
    if not tts_model:
        print("Error: no TTS model found. Use --tts-model.", file=sys.stderr)
        sys.exit(1)

    print(f"Models: llm={llm_model}  stt={stt_model}  tts={tts_model}  voice={args.voice}")
    print(f"Mode:   {args.mode}\n")

    kwargs = dict(
        audio=audio,
        text=args.text,
        base_url=args.base_url,
        llm_model=llm_model,
        stt_model=stt_model or "",
        tts_model=tts_model,
        voice=args.voice,
        instructions=args.instructions,
        output_path=args.output,
    )

    if args.mode == "sequential":
        await run_sequential(**kwargs)
    else:
        await run_pipelined(**kwargs)


if __name__ == "__main__":
    asyncio.run(main())

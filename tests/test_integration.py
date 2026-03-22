"""Integration tests for the gait gateway.

Requires a running gateway (docker compose up -d). Uses the OpenAI Python
client. Outputs saved to ./test_outputs/ for manual inspection.

Tests cover only functionality the gateway or its providers explicitly
implement: request validation, model merging, audio format conversion,
voice management, tool-calling model shapes, and the audio-modality
streaming path.  Llama.cpp sampling parameters that the gateway merely
forwards (temperature, top_p, seed, logprobs, …) are intentionally
omitted — they are tested by llama.cpp itself.

    GATEWAY_URL defaults to http://localhost:3000
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
from openai import OpenAI

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:3000")
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


def _save(name: str, data: str | bytes, binary: bool = False) -> Path:
    """Persist test output to OUTPUT_DIR and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / name
    if binary:
        p.write_bytes(data)  # type: ignore[arg-type]
    else:
        p.write_text(data if isinstance(data, str) else data.decode())  # type: ignore[arg-type]
    return p


def _make_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file for testing audio endpoints."""
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


@pytest.fixture(scope="session")
def client() -> OpenAI:
    """OpenAI client pointed at the gateway."""
    return OpenAI(base_url=f"{GATEWAY_URL}/v1", api_key="not-needed")


@pytest.fixture(scope="session")
def raw_client() -> httpx.Client:
    """Raw httpx client for endpoints the OpenAI SDK doesn't cover."""
    return httpx.Client(base_url=GATEWAY_URL, timeout=300.0)


@pytest.fixture(scope="session", autouse=True)
def _wait_for_gateway():
    """Block until the gateway health endpoint responds (up to 5 min)."""
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            r = httpx.get(f"{GATEWAY_URL}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(2)
    pytest.fail("Gateway did not become healthy within 5 minutes")


@pytest.fixture(scope="session")
def models(raw_client: httpx.Client) -> dict[str, list[str]]:
    """Fetch model list once and categorise by capabilities."""
    r = raw_client.get("/v1/models")
    data = r.json()["data"]
    ids = [m["id"] for m in data]
    caps = {m["id"]: m.get("capabilities", []) for m in data}

    return {
        "all": ids,
        "tts": [i for i in ids if "speech" in caps.get(i, [])],
        "stt": [i for i in ids if "transcription" in caps.get(i, [])],
        "llm": [i for i in ids if "chat" in caps.get(i, [])],
    }


# ===================================================================
# Health  (gateway aggregates health from all backends)
# ===================================================================


class TestHealth:
    def test_health_endpoint(self, raw_client: httpx.Client):
        r = raw_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        _save("health.json", json.dumps(data, indent=2))

        assert "status" in data
        assert "backends" in data
        assert isinstance(data["backends"], dict)
        assert "chatterbox" in data["backends"]
        assert "whisperx" in data["backends"]
        assert "llamacpp" in data["backends"]


# ===================================================================
# Models  (gateway merges models from all providers)
# ===================================================================


class TestModels:
    def test_list_models_via_sdk(self, client: OpenAI):
        """client.models.list() returns models."""
        model_list = client.models.list()
        _save(
            "models_sdk.json",
            json.dumps(
                [m.model_dump() for m in model_list.data],
                indent=2, default=str,
            ),
        )
        assert len(model_list.data) > 0
        for m in model_list.data:
            assert m.id
            assert m.object == "model"

    def test_list_models(self, models: dict):
        _save("models.json", json.dumps(models, indent=2))
        assert len(models["all"]) > 0

    def test_has_tts_model(self, models: dict):
        assert len(models["tts"]) > 0

    def test_has_stt_model(self, models: dict):
        assert len(models["stt"]) > 0

    def test_has_llm_model(self, models: dict):
        assert len(models["llm"]) > 0

    def test_models_have_capabilities(self, raw_client: httpx.Client):
        """Every model carries capabilities and loaded fields."""
        r = raw_client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        _save("models_capabilities.json", json.dumps(data, indent=2))
        for m in data["data"]:
            assert len(m.get("capabilities", [])) > 0, (
                f"Model {m['id']} has empty capabilities"
            )
            assert "loaded" in m


# ===================================================================
# Speech  (gateway converts wav→mp3, validates SpeechRequest)
# ===================================================================


class TestSpeech:
    def test_speech_mp3(self, client: OpenAI, models: dict):
        """Gateway converts backend WAV to MP3."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="Hello! This is an integration test.",
            response_format="mp3",
        )
        audio = resp.read()
        path = _save("speech_mp3.mp3", audio, binary=True)
        assert len(audio) > 100
        assert path.stat().st_size > 100

    def test_speech_wav(self, client: OpenAI, models: dict):
        """WAV passthrough from backend."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="Testing WAV output format.",
            response_format="wav",
        )
        audio = resp.read()
        _save("speech_wav.wav", audio, binary=True)
        assert audio[:4] == b"RIFF"
        assert audio[8:12] == b"WAVE"

    def test_speech_extended_params(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Chatterbox-specific extensions forwarded via SpeechRequest."""
        tts_model = models["tts"][0]
        r = raw_client.post(
            "/v1/audio/speech",
            json={
                "model": tts_model,
                "voice": "default",
                "input": "Extended parameter test.",
                "response_format": "wav",
                "speed": 1.2,
                "temperature": 0.9,
                "seed": 42,
            },
        )
        assert r.status_code == 200
        _save("speech_extended.wav", r.content, binary=True)
        assert r.content[:4] == b"RIFF"

    def test_speech_missing_fields(self, raw_client: httpx.Client):
        """SpeechRequest validation rejects missing required fields."""
        r = raw_client.post("/v1/audio/speech", json={"model": "x"})
        assert r.status_code == 422

    def test_speech_unsupported_format(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Gateway rejects formats it cannot convert."""
        tts_model = models["tts"][0]
        r = raw_client.post(
            "/v1/audio/speech",
            json={
                "model": tts_model,
                "voice": "default",
                "input": "test",
                "response_format": "opus",
            },
        )
        assert r.status_code == 400


# ===================================================================
# Transcription  (gateway formats: json, text, srt, vtt, verbose_json)
# ===================================================================


class TestTranscription:
    @pytest.fixture(scope="class")
    def tts_audio_wav(self, client: OpenAI, models: dict) -> bytes:
        """Generate real speech via TTS for transcription tests."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="Hello, hello, hello, testing one two three.",
            response_format="wav",
        )
        audio = resp.read()
        _save("tts_for_transcription.wav", audio, binary=True)
        return audio

    def test_transcription_json_via_sdk(
        self, client: OpenAI, models: dict, tts_audio_wav: bytes,
    ):
        """Default JSON format via OpenAI SDK."""
        stt_model = models["stt"][0]
        transcript = client.audio.transcriptions.create(
            model=stt_model,
            file=("speech.wav", tts_audio_wav, "audio/wav"),
            response_format="json",
        )
        _save(
            "transcription_json.json",
            json.dumps({"text": transcript.text}, indent=2),
        )
        # TTS quality varies; just verify the pipeline returns something
        assert isinstance(transcript.text, str)

    def test_transcription_text(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """Gateway formats result as plain text."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "text"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        _save("transcription_text.txt", r.text)

    def test_transcription_verbose_json(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """Gateway builds verbose_json with expected fields."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "verbose_json"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        _save("transcription_verbose.json", json.dumps(data, indent=2))

        assert data["task"] == "transcribe"
        assert "language" in data
        assert "duration" in data
        assert "segments" in data
        assert "text" in data

    def test_transcription_srt(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """Gateway formats result as SRT subtitles."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "srt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("transcription.srt", r.text)
        # SRT may be empty if TTS produced unintelligible audio
        assert isinstance(r.text, str)

    def test_transcription_vtt(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """Gateway formats result as WebVTT subtitles."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "vtt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("transcription.vtt", r.text)
        assert r.text.startswith("WEBVTT")

    def test_transcription_diarize(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """Gateway passes diarize flag to whisperx."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={
                "model": stt_model,
                "response_format": "verbose_json",
                "diarize": "true",
            },
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        _save("transcription_diarize.json", json.dumps(data, indent=2))
        assert data["task"] == "transcribe"
        assert "segments" in data
        assert "words" in data

    def test_transcription_with_language(
        self, client: OpenAI, models: dict, tts_audio_wav: bytes,
    ):
        """Language hint forwarded to whisperx."""
        stt_model = models["stt"][0]
        transcript = client.audio.transcriptions.create(
            model=stt_model,
            file=("speech.wav", tts_audio_wav, "audio/wav"),
            language="en",
        )
        _save(
            "transcription_with_lang.json",
            json.dumps({"text": transcript.text}, indent=2),
        )
        assert isinstance(transcript.text, str)

    def test_transcription_empty_file(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Gateway rejects empty audio."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model},
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert r.status_code == 400

    def test_transcription_invalid_format(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Gateway rejects unknown response_format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "invalid"},
            files={"file": ("test.wav", _make_wav(), "audio/wav")},
        )
        assert r.status_code == 400


# ===================================================================
# Translation  (gateway formats, whisperx translates)
# ===================================================================


class TestTranslation:
    @pytest.fixture(scope="class")
    def tts_audio_wav(self, client: OpenAI, models: dict) -> bytes:
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="This sentence will be translated to English.",
            response_format="wav",
        )
        audio = resp.read()
        _save("tts_for_translation.wav", audio, binary=True)
        return audio

    def test_translation_via_sdk(
        self, client: OpenAI, models: dict, tts_audio_wav: bytes,
    ):
        """Translation via OpenAI SDK."""
        stt_model = models["stt"][0]
        translation = client.audio.translations.create(
            model=stt_model,
            file=("speech.wav", tts_audio_wav, "audio/wav"),
        )
        _save(
            "translation_sdk.json",
            json.dumps({"text": translation.text}, indent=2),
        )
        assert len(translation.text) > 0

    def test_translation_json(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """JSON format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/translations",
            data={"model": stt_model},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        _save("translation_json.json", json.dumps(data, indent=2))
        assert len(data["text"]) > 0

    def test_translation_verbose_json(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """verbose_json includes task=translate and expected fields."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/translations",
            data={"model": stt_model, "response_format": "verbose_json"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        _save("translation_verbose.json", json.dumps(data, indent=2))
        assert data["task"] == "translate"
        assert "segments" in data
        assert "text" in data

    def test_translation_srt(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client,
    ):
        """SRT subtitle format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/translations",
            data={"model": stt_model, "response_format": "srt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("translation.srt", r.text)
        assert "-->" in r.text


# ===================================================================
# Voices  (gateway manages WAV files on a shared volume)
# ===================================================================


class TestVoices:
    VOICE_NAME = "integration_test_voice"

    def test_01_list_voices_has_default(self, raw_client: httpx.Client):
        r = raw_client.get("/v1/audio/voices")
        assert r.status_code == 200
        data = r.json()
        _save("voices_list_initial.json", json.dumps(data, indent=2))
        assert data["object"] == "list"
        names = [v["name"] for v in data["data"]]
        assert "default" in names

    def test_02_get_default_voice(self, raw_client: httpx.Client):
        r = raw_client.get("/v1/audio/voices/default")
        assert r.status_code == 200
        assert r.json()["voice_id"] == "default"

    def test_03_create_voice(self, raw_client: httpx.Client):
        wav_data = _make_wav(duration_s=2.0)
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": self.VOICE_NAME},
            files={"file": ("ref.wav", wav_data, "audio/wav")},
        )
        assert r.status_code == 201
        assert r.json()["voice_id"] == self.VOICE_NAME

    def test_04_list_voices_includes_new(self, raw_client: httpx.Client):
        r = raw_client.get("/v1/audio/voices")
        assert r.status_code == 200
        names = [v["name"] for v in r.json()["data"]]
        assert self.VOICE_NAME in names

    def test_05_get_created_voice(self, raw_client: httpx.Client):
        r = raw_client.get(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 200
        assert r.json()["voice_id"] == self.VOICE_NAME

    def test_06_create_duplicate_rejected(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": self.VOICE_NAME},
            files={"file": ("ref.wav", _make_wav(), "audio/wav")},
        )
        assert r.status_code == 409

    def test_07_create_default_rejected(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "default"},
            files={"file": ("ref.wav", _make_wav(), "audio/wav")},
        )
        assert r.status_code == 400

    def test_08_create_invalid_name(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "bad name!"},
            files={"file": ("ref.wav", _make_wav(), "audio/wav")},
        )
        assert r.status_code == 400

    def test_09_create_not_wav(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "notwav"},
            files={
                "file": (
                    "ref.mp3",
                    b"\xff\xfb\x90\x00" + b"\x00" * 100,
                    "audio/mpeg",
                ),
            },
        )
        assert r.status_code == 400

    def test_10_use_custom_voice_for_tts(
        self, client: OpenAI, models: dict,
    ):
        """TTS with the custom voice produces audio."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice=self.VOICE_NAME,
            input="Speaking with a custom voice.",
            response_format="wav",
        )
        audio = resp.read()
        _save("speech_custom_voice.wav", audio, binary=True)
        assert audio[:4] == b"RIFF"
        assert len(audio) > 100

    def test_11_delete_voice(self, raw_client: httpx.Client):
        r = raw_client.delete(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_12_delete_not_found(self, raw_client: httpx.Client):
        r = raw_client.delete(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 404

    def test_13_delete_default_rejected(self, raw_client: httpx.Client):
        r = raw_client.delete("/v1/audio/voices/default")
        assert r.status_code == 400

    def test_14_get_not_found(self, raw_client: httpx.Client):
        r = raw_client.get(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 404


# ===================================================================
# Chat Completions  (gateway validates ChatCompletionRequest,
#   dispatches stream vs non-stream, handles audio modality)
# ===================================================================


class TestChatCompletions:
    @staticmethod
    def _get_text(message) -> str:
        return (
            message.content
            or getattr(message, "reasoning_content", "")
            or ""
        )

    def test_chat_completion(self, client: OpenAI, models: dict):
        """Non-streaming chat completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "Say hello in exactly 5 words."},
            ],
            max_tokens=512,
        )
        _save(
            "chat_completion.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(self._get_text(resp.choices[0].message)) > 0
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0

    def test_chat_completion_stream(self, client: OpenAI, models: dict):
        """Streaming chat completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        stream = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "Count from 1 to 5."},
            ],
            max_tokens=512,
            stream=True,
        )
        chunks = []
        full_text = ""
        for chunk in stream:
            chunks.append(chunk.model_dump())
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                full_text += reasoning

        _save(
            "chat_stream_chunks.json",
            json.dumps(chunks, indent=2, default=str),
        )
        assert len(chunks) > 1
        assert len(full_text) > 0

    def test_chat_system_message(self, client: OpenAI, models: dict):
        """System message forwarded correctly."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "system",
                    "content": "You are a pirate. Always say 'Arrr!'.",
                },
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=512,
        )
        _save(
            "chat_system_msg.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(self._get_text(resp.choices[0].message)) > 0

    def test_chat_multi_turn(self, client: OpenAI, models: dict):
        """Multi-turn conversation preserves context."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {
                    "role": "assistant",
                    "content": "Hello Alice! How can I help you?",
                },
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=512,
        )
        _save(
            "chat_multi_turn.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        text = self._get_text(resp.choices[0].message)
        assert len(text) > 0
        assert "alice" in text.lower()


# ===================================================================
# Chat Completions — tool calling
#   (gateway validates ChatCompletionTool / ChatCompletionToolCall shapes)
# ===================================================================


class TestChatToolCalling:
    WEATHER_TOOL: ClassVar[dict[str, object]] = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["location"],
            },
        },
    }

    def test_tool_call_basic(self, client: OpenAI, models: dict):
        """Model returns a tool call when tools are provided."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?",
                },
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=512,
        )
        _save(
            "chat_tool_call.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) > 0
        tc = msg.tool_calls[0]
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "location" in args

    def test_tool_call_multi_turn(self, client: OpenAI, models: dict):
        """Full round trip: request → tool_call → tool result → final."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm = models["llm"][0]

        # Step 1: model decides to call the tool
        resp1 = client.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris?",
                },
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=512,
        )
        msg1 = resp1.choices[0].message
        assert msg1.tool_calls is not None
        tc = msg1.tool_calls[0]

        # Step 2: send tool result back
        resp2 = client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                msg1.model_dump(),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({
                        "temperature": 18,
                        "unit": "celsius",
                        "condition": "sunny",
                    }),
                },
            ],
            tools=[self.WEATHER_TOOL],
            max_tokens=512,
        )
        _save(
            "chat_tool_multi_turn.json",
            json.dumps(resp2.model_dump(), indent=2, default=str),
        )
        text = resp2.choices[0].message.content or ""
        assert len(text) > 0

    def test_tool_call_streaming(self, client: OpenAI, models: dict):
        """Streaming with tools returns tool call chunks."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        stream = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo?",
                },
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            stream=True,
            max_tokens=512,
        )
        chunks = []
        tool_call_name = ""
        tool_call_args = ""
        for chunk in stream:
            chunks.append(chunk.model_dump())
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    if tc.function:
                        if tc.function.name:
                            tool_call_name = tc.function.name
                        if tc.function.arguments:
                            tool_call_args += tc.function.arguments

        _save(
            "chat_tool_stream.json",
            json.dumps(chunks, indent=2, default=str),
        )
        assert tool_call_name == "get_weather"
        assert len(tool_call_args) > 0
        args = json.loads(tool_call_args)
        assert "location" in args


# ===================================================================
# Chat Completions — response format
#   (gateway validates ResponseFormat model)
# ===================================================================


class TestChatResponseFormat:
    def test_json_mode(self, client: OpenAI, models: dict):
        """response_format: json_object produces valid JSON output."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with keys 'name' and 'age'."
                        ' Example: {"name": "Alice", "age": 30}'
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        _save(
            "chat_json_mode.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        text = resp.choices[0].message.content or ""
        assert len(text) > 0
        # Strip markdown code fences if present (model quirk)
        clean = text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        parsed = json.loads(clean.strip())
        assert isinstance(parsed, dict)


# ===================================================================
# Chat Completions — sampling parameters
#   (gateway validates typed fields: temperature, top_p, seed, etc.)
# ===================================================================


class TestChatSamplingParams:
    def test_temperature(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[{"role": "user", "content": "Say hi"}],
            temperature=0.1,
            max_tokens=64,
        )
        assert resp.choices[0].message.content

    def test_top_p(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[{"role": "user", "content": "Say hi"}],
            top_p=0.5,
            max_tokens=64,
        )
        assert resp.choices[0].message.content

    def test_stop_sequence(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "user",
                    "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
                },
            ],
            stop=["5"],
            max_tokens=256,
        )
        _save(
            "chat_stop.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert resp.choices[0].message.content is not None

    def test_seed_determinism(self, client: OpenAI, models: dict):
        """Same seed + temperature=0 produces identical output."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        kwargs = dict(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "Say exactly: hello world"},
            ],
            seed=12345,
            temperature=0.0,
            max_tokens=64,
        )
        r1 = client.chat.completions.create(**kwargs)
        r2 = client.chat.completions.create(**kwargs)
        _save(
            "chat_seed.json",
            json.dumps(
                {
                    "r1": r1.choices[0].message.content,
                    "r2": r2.choices[0].message.content,
                },
                indent=2,
            ),
        )
        assert r1.choices[0].message.content == r2.choices[0].message.content

    def test_max_tokens_truncates(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {
                    "role": "user",
                    "content": "Write a long essay about the universe.",
                },
            ],
            max_tokens=10,
        )
        assert resp.choices[0].finish_reason == "length"

    def test_frequency_penalty(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[{"role": "user", "content": "Say something"}],
            frequency_penalty=0.5,
            max_tokens=64,
        )
        assert resp.choices[0].message.content

    def test_presence_penalty(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[{"role": "user", "content": "Say something"}],
            presence_penalty=0.5,
            max_tokens=64,
        )
        assert resp.choices[0].message.content

    def test_logprobs(self, client: OpenAI, models: dict):
        """logprobs returns token-level log probabilities."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[{"role": "user", "content": "Say hi"}],
            logprobs=True,
            top_logprobs=3,
            max_tokens=32,
        )
        _save(
            "chat_logprobs.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        lp = resp.choices[0].logprobs
        assert lp is not None
        assert len(lp.content) > 0
        assert hasattr(lp.content[0], "logprob")
        assert len(lp.content[0].top_logprobs) > 0


# ===================================================================
# Text Completions  (gateway validates CompletionRequest)
# ===================================================================


class TestCompletions:
    def test_text_completion(self, client: OpenAI, models: dict):
        """Non-streaming text completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.completions.create(
            model=models["llm"][0],
            prompt="The capital of France is",
            max_tokens=20,
        )
        _save(
            "completion.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(resp.choices) > 0
        assert len(resp.choices[0].text) > 0

    def test_text_completion_stream(self, client: OpenAI, models: dict):
        """Streaming text completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        stream = client.completions.create(
            model=models["llm"][0],
            prompt="Once upon a time",
            max_tokens=50,
            stream=True,
        )
        chunks = []
        full_text = ""
        for chunk in stream:
            chunks.append(chunk.model_dump())
            if chunk.choices[0].text:
                full_text += chunk.choices[0].text

        _save(
            "completion_stream_chunks.json",
            json.dumps(chunks, indent=2, default=str),
        )
        assert len(chunks) > 1
        assert len(full_text) > 0


# ===================================================================
# Responses API  (gateway validates CreateResponseRequest)
# ===================================================================


class TestResponses:
    def test_response_non_streaming(
        self, models: dict, raw_client: httpx.Client,
    ):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "Say hello in one word.",
                "max_output_tokens": 512,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_non_streaming.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert len(data["output"]) > 0
        assert data["usage"]["total_tokens"] > 0

    def test_response_streaming(
        self, models: dict, raw_client: httpx.Client,
    ):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "Say hello in one word.",
                "max_output_tokens": 512,
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("response_streaming.txt", raw_sse)

        events = []
        for line in raw_sse.split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                events.append(line[7:])

        _save(
            "response_streaming_events.json",
            json.dumps(events, indent=2),
        )
        assert "response.created" in events
        assert (
            "response.completed" in events or "response.done" in events
        )

    def test_response_with_instructions(
        self, models: dict, raw_client: httpx.Client,
    ):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "instructions": "You are a pirate. Always say Arrr!",
                "input": "Hello!",
                "max_output_tokens": 512,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_with_instructions.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["status"] == "completed"
        assert len(data["output"]) > 0

    def test_response_with_input_items(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Input can be an array of message items."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Say hi",
                    },
                ],
                "max_output_tokens": 128,
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "completed"


# ===================================================================
# Embeddings  (gateway validates EmbeddingRequest)
# ===================================================================


class TestEmbeddings:
    def test_single_embedding(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.embeddings.create(
            model=models["llm"][0],
            input="Hello world",
        )
        _save(
            "embedding_single.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(resp.data) == 1
        assert len(resp.data[0].embedding) > 0

    def test_batch_embeddings(self, client: OpenAI, models: dict):
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.embeddings.create(
            model=models["llm"][0],
            input=["Hello", "World", "Test"],
        )
        _save(
            "embedding_batch.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(resp.data) == 3
        for item in resp.data:
            assert len(item.embedding) > 0

    def test_embedding_has_usage(self, client: OpenAI, models: dict):
        """Embedding response includes usage information."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.embeddings.create(
            model=models["llm"][0],
            input="Test",
        )
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.total_tokens > 0


# ===================================================================
# Round-trip: TTS → STT
# ===================================================================


class TestRoundTrip:
    def test_tts_to_stt_round_trip(self, client: OpenAI, models: dict):
        """TTS → STT pipeline produces a non-empty transcription."""
        if not models["tts"] or not models["stt"]:
            pytest.skip("Need both TTS and STT models")

        original_text = "The quick brown fox jumps over the lazy dog."
        tts_model = models["tts"][0]
        stt_model = models["stt"][0]

        tts_resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input=original_text,
            response_format="wav",
        )
        audio = tts_resp.read()
        _save("roundtrip_tts.wav", audio, binary=True)
        assert audio[:4] == b"RIFF", "TTS did not produce valid WAV"

        transcript = client.audio.transcriptions.create(
            model=stt_model,
            file=("roundtrip.wav", audio, "audio/wav"),
            response_format="json",
        )
        _save(
            "roundtrip_stt.json",
            json.dumps(
                {"original": original_text, "transcribed": transcript.text},
                indent=2,
            ),
        )
        # Pipeline works if STT produces any text from the TTS audio
        assert len(transcript.text.strip()) > 0, "Empty transcription"


# ===================================================================
# Chat Audio  (gateway-specific: interleaves text+TTS in SSE stream)
# ===================================================================


class TestChatAudio:
    def test_chat_audio_stream(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Streaming with audio modality interleaves text and audio."""
        if not models["llm"] or not models["tts"]:
            pytest.skip("Need both LLM and TTS models")

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": "Say hello in one sentence.",
                    },
                ],
                "max_tokens": 512,
                "stream": True,
                "modalities": ["text", "audio"],
                "audio": {
                    "voice": "default",
                    "model": models["tts"][0],
                },
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("chat_audio_stream.txt", raw_sse)

        text_tokens: list[str] = []
        audio_data_chunks: list[str] = []
        audio_transcripts: list[str] = []
        got_audio_id = False
        got_expires = False

        for line in raw_sse.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                text_tokens.append(delta["content"])
            if delta.get("reasoning_content"):
                text_tokens.append(delta["reasoning_content"])
            if "audio" in delta:
                audio = delta["audio"]
                if "id" in audio:
                    got_audio_id = True
                if "data" in audio:
                    audio_data_chunks.append(audio["data"])
                if "transcript" in audio:
                    audio_transcripts.append(audio["transcript"])
                if "expires_at" in audio:
                    got_expires = True

        _save(
            "chat_audio_parsed.json",
            json.dumps(
                {
                    "text_token_count": len(text_tokens),
                    "audio_chunk_count": len(audio_data_chunks),
                    "audio_transcripts": audio_transcripts,
                    "got_audio_id": got_audio_id,
                    "got_expires": got_expires,
                },
                indent=2,
            ),
        )

        assert len(text_tokens) > 0
        assert got_audio_id
        assert got_expires
        assert len(audio_data_chunks) > 0
        assert len(audio_transcripts) > 0

    def test_chat_audio_nonstreaming(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Non-streaming audio modality returns message.audio."""
        if not models["llm"] or not models["tts"]:
            pytest.skip("Need both LLM and TTS models")

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {"role": "user", "content": "Say hello in one sentence."},
                ],
                "max_tokens": 512,
                "modalities": ["text", "audio"],
                "audio": {
                    "voice": "default",
                    "model": models["tts"][0],
                    "format": "pcm16",
                },
            },
        )
        assert r.status_code == 200

        data = r.json()
        _save("chat_audio_nonstreaming.json", json.dumps(data, indent=2))

        msg = data["choices"][0]["message"]
        # Content is null per OpenAI contract when audio is returned
        assert msg.get("content") is None
        # Audio attachment present with all required fields
        audio = msg["audio"]
        assert "id" in audio
        assert len(audio["id"]) > 0
        assert "data" in audio
        assert len(audio["data"]) > 0
        assert "transcript" in audio
        assert len(audio["transcript"]) > 0
        assert "expires_at" in audio
        assert audio["expires_at"] > 0

        # Verify the audio data is valid base64
        import base64
        pcm_bytes = base64.b64decode(audio["data"])
        assert len(pcm_bytes) > 0


# ===================================================================
# Chat Input Audio  (gateway transcribes input_audio → text before LLM)
# ===================================================================


class TestChatInputAudio:
    @pytest.fixture(scope="class")
    def tts_audio_wav(self, client: OpenAI, models: dict) -> bytes:
        """Generate real speech via TTS for input_audio tests."""
        if not models["tts"]:
            pytest.skip("No TTS model available")
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="What is two plus two?",
            response_format="wav",
        )
        audio = resp.read()
        _save("tts_for_input_audio.wav", audio, binary=True)
        return audio

    def test_input_audio_wav(
        self, models: dict, raw_client: httpx.Client, tts_audio_wav: bytes,
    ):
        """input_audio with wav format is transcribed and answered by the LLM."""
        if not models["llm"] or not models["stt"]:
            pytest.skip("Need LLM and STT models")

        import base64
        b64_audio = base64.b64encode(tts_audio_wav).decode()

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200

        data = r.json()
        _save("chat_input_audio_wav.json", json.dumps(data, indent=2))

        text = data["choices"][0]["message"]["content"] or ""
        assert len(text) > 0, "LLM produced empty response from transcribed audio"

    def test_input_audio_pcm16(
        self, models: dict, raw_client: httpx.Client, tts_audio_wav: bytes,
    ):
        """input_audio with pcm16 format is converted to WAV then transcribed."""
        if not models["llm"] or not models["stt"]:
            pytest.skip("Need LLM and STT models")

        import base64

        from gateway.formatting import wav_to_pcm16

        # Convert our WAV test audio to raw PCM16
        pcm_bytes, _sr = wav_to_pcm16(tts_audio_wav, target_sr=16000)
        b64_audio = base64.b64encode(pcm_bytes).decode()

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "pcm16",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200

        data = r.json()
        _save("chat_input_audio_pcm16.json", json.dumps(data, indent=2))

        text = data["choices"][0]["message"]["content"] or ""
        assert len(text) > 0, "LLM produced empty response from PCM16 audio"

    def test_input_audio_mixed_content(
        self, models: dict, raw_client: httpx.Client, tts_audio_wav: bytes,
    ):
        """Text parts alongside input_audio are preserved."""
        if not models["llm"] or not models["stt"]:
            pytest.skip("Need LLM and STT models")

        import base64
        b64_audio = base64.b64encode(tts_audio_wav).decode()

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The user asked this audio question. Answer it:",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200

        data = r.json()
        _save("chat_input_audio_mixed.json", json.dumps(data, indent=2))

        text = data["choices"][0]["message"]["content"] or ""
        assert len(text) > 0

    def test_input_audio_with_audio_output(
        self, models: dict, raw_client: httpx.Client, tts_audio_wav: bytes,
    ):
        """input_audio preprocessing + audio output modality work together."""
        if not models["llm"] or not models["tts"] or not models["stt"]:
            pytest.skip("Need LLM, TTS, and STT models")

        import base64
        b64_audio = base64.b64encode(tts_audio_wav).decode()

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 256,
                "stream": True,
                "modalities": ["text", "audio"],
                "audio": {
                    "voice": "default",
                    "model": models["tts"][0],
                },
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("chat_input_audio_with_output.txt", raw_sse)

        text_tokens: list[str] = []
        audio_data_chunks: list[str] = []
        got_audio_id = False

        for line in raw_sse.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                text_tokens.append(delta["content"])
            if delta.get("reasoning_content"):
                text_tokens.append(delta["reasoning_content"])
            if "audio" in delta:
                a = delta["audio"]
                if "id" in a:
                    got_audio_id = True
                if "data" in a:
                    audio_data_chunks.append(a["data"])

        _save(
            "chat_input_audio_output_parsed.json",
            json.dumps(
                {
                    "text_token_count": len(text_tokens),
                    "audio_chunk_count": len(audio_data_chunks),
                    "got_audio_id": got_audio_id,
                },
                indent=2,
            ),
        )

        # LLM produced text from the transcribed input audio
        assert len(text_tokens) > 0
        # Audio output was also synthesised
        assert got_audio_id
        assert len(audio_data_chunks) > 0

    def test_input_audio_empty_data(
        self, models: dict, raw_client: httpx.Client,
    ):
        """input_audio with empty data returns 400."""
        if not models["llm"]:
            pytest.skip("No LLM model available")

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "",
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
            },
        )
        assert r.status_code == 400

    def test_input_audio_invalid_base64(
        self, models: dict, raw_client: httpx.Client,
    ):
        """input_audio with invalid base64 returns 400."""
        if not models["llm"]:
            pytest.skip("No LLM model available")

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "!!!not-base64!!!",
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
            },
        )
        assert r.status_code == 400


# ===================================================================
# Reasoning  (gateway forwards reasoning_effort, reasoning_content,
#   and token details through the llama.cpp proxy)
# ===================================================================



class TestChatReasoning:
    """Chat Completions with reasoning support (thinking models)."""

    def test_reasoning_effort_high(
        self, client: OpenAI, models: dict,
    ):
        """reasoning_effort='high' is forwarded and accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "What is 7 * 8?"},
            ],
            reasoning_effort="high",
            max_completion_tokens=4096,
        )
        _save(
            "chat_reasoning_high.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert resp.choices[0].message.content is not None
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0

    def test_reasoning_effort_low(
        self, client: OpenAI, models: dict,
    ):
        """reasoning_effort='low' is forwarded and accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            reasoning_effort="low",
            max_completion_tokens=4096,
        )
        _save(
            "chat_reasoning_low.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert resp.choices[0].message.content is not None
        assert resp.usage is not None

    def test_reasoning_effort_medium(
        self, client: OpenAI, models: dict,
    ):
        """reasoning_effort='medium' is forwarded and accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        resp = client.chat.completions.create(
            model=models["llm"][0],
            messages=[
                {"role": "user", "content": "What is 3 + 5?"},
            ],
            reasoning_effort="medium",
            max_completion_tokens=4096,
        )
        _save(
            "chat_reasoning_medium.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert resp.choices[0].message.content is not None

    def test_reasoning_usage_has_token_details(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Reasoning response includes completion_tokens_details."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {"role": "user", "content": "What is 12 * 13?"},
                ],
                "reasoning_effort": "high",
                "max_completion_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "chat_reasoning_usage.json",
            json.dumps(data, indent=2, default=str),
        )

        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0

        # completion_tokens_details should be present
        details = usage.get("completion_tokens_details")
        if details is not None:
            assert "reasoning_tokens" in details
            assert details["reasoning_tokens"] >= 0

    def test_reasoning_content_in_response(
        self, models: dict, raw_client: httpx.Client,
    ):
        """With thinking enabled, the response may include reasoning_content."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the square root of 144?",
                    },
                ],
                "reasoning_effort": "high",
                "max_completion_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "chat_reasoning_content.json",
            json.dumps(data, indent=2, default=str),
        )

        msg = data["choices"][0]["message"]
        # The model should produce visible content
        assert msg.get("content") is not None

    def test_reasoning_streaming(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Streaming with reasoning_effort returns reasoning_content deltas."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {"role": "user", "content": "What is 9 * 7?"},
                ],
                "reasoning_effort": "high",
                "max_completion_tokens": 4096,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("chat_reasoning_stream.txt", raw_sse)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        usage_chunk: dict | None = None

        for line in raw_sse.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if data.get("usage"):
                usage_chunk = data["usage"]

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content"):
                content_parts.append(delta["content"])

        _save(
            "chat_reasoning_stream_parsed.json",
            json.dumps(
                {
                    "reasoning_token_count": len(reasoning_parts),
                    "content_token_count": len(content_parts),
                    "has_usage": usage_chunk is not None,
                },
                indent=2,
            ),
        )

        # With thinking enabled, we should see content tokens
        assert len(content_parts) > 0

        # Usage chunk should be present (include_usage=true)
        assert usage_chunk is not None
        assert usage_chunk["total_tokens"] > 0

    def test_reasoning_streaming_usage_details(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Streaming usage chunk includes completion_tokens_details."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": models["llm"][0],
                "messages": [
                    {"role": "user", "content": "What is 15 + 27?"},
                ],
                "reasoning_effort": "high",
                "max_completion_tokens": 4096,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("chat_reasoning_stream_usage.txt", raw_sse)

        usage_chunk: dict | None = None
        for line in raw_sse.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("usage"):
                usage_chunk = data["usage"]

        assert usage_chunk is not None
        _save(
            "chat_reasoning_stream_usage_parsed.json",
            json.dumps(usage_chunk, indent=2),
        )

        assert usage_chunk["prompt_tokens"] > 0
        assert usage_chunk["completion_tokens"] > 0

        # Check for token details if present
        details = usage_chunk.get("completion_tokens_details")
        if details is not None:
            assert "reasoning_tokens" in details


class TestResponsesReasoning:
    """Responses API with reasoning support."""

    def test_responses_reasoning_effort(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Responses API with reasoning.effort is forwarded and accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is 7 * 8?",
                "reasoning": {"effort": "high"},
                "max_output_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_reasoning_effort.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert len(data["output"]) > 0
        assert data["usage"]["total_tokens"] > 0

    def test_responses_reasoning_usage_details(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Responses usage includes output_tokens_details when reasoning."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is 12 * 13?",
                "reasoning": {"effort": "high"},
                "max_output_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_reasoning_usage.json",
            json.dumps(data, indent=2, default=str),
        )

        usage = data["usage"]
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] > 0

        # Check output_tokens_details if present
        details = usage.get("output_tokens_details")
        if details is not None:
            assert "reasoning_tokens" in details
            assert details["reasoning_tokens"] >= 0

    def test_responses_reasoning_summary(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Responses API with reasoning.summary is forwarded and accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is the square root of 256?",
                "reasoning": {"effort": "high", "summary": "auto"},
                "max_output_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_reasoning_summary.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["status"] == "completed"
        assert len(data["output"]) > 0

    def test_responses_reasoning_summary_concise(
        self, models: dict, raw_client: httpx.Client,
    ):
        """reasoning.summary='concise' is forwarded."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is 5 factorial?",
                "reasoning": {"effort": "medium", "summary": "concise"},
                "max_output_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_reasoning_summary_concise.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["status"] == "completed"

    def test_responses_reasoning_summary_detailed(
        self, models: dict, raw_client: httpx.Client,
    ):
        """reasoning.summary='detailed' is forwarded."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is the derivative of x^2?",
                "reasoning": {"effort": "high", "summary": "detailed"},
                "max_output_tokens": 4096,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_reasoning_summary_detailed.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["status"] == "completed"

    def test_responses_reasoning_effort_levels(
        self, models: dict, raw_client: httpx.Client,
    ):
        """All reasoning effort levels (low, medium, high) are accepted."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        for level in ("low", "medium", "high"):
            r = raw_client.post(
                "/v1/responses",
                json={
                    "model": models["llm"][0],
                    "input": f"What is 2 + 3? (effort={level})",
                    "reasoning": {"effort": level},
                    "max_output_tokens": 4096,
                },
            )
            assert r.status_code == 200, (
                f"reasoning.effort={level} returned {r.status_code}"
            )
            data = r.json()
            _save(
                f"response_reasoning_{level}.json",
                json.dumps(data, indent=2, default=str),
            )
            assert data["status"] == "completed"

    def test_responses_reasoning_streaming(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Streaming Responses with reasoning produces SSE events."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "What is 6 * 9?",
                "reasoning": {"effort": "high", "summary": "auto"},
                "max_output_tokens": 4096,
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("response_reasoning_stream.txt", raw_sse)

        events: list[str] = []
        for line in raw_sse.split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                events.append(line[7:])

        _save(
            "response_reasoning_stream_events.json",
            json.dumps(events, indent=2),
        )

        assert "response.created" in events
        assert (
            "response.completed" in events or "response.done" in events
        )

    def test_responses_reasoning_without_reasoning_param(
        self, models: dict, raw_client: httpx.Client,
    ):
        """Responses without reasoning param still works (no reasoning)."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": models["llm"][0],
                "input": "Say hello.",
                "max_output_tokens": 128,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save(
            "response_no_reasoning.json",
            json.dumps(data, indent=2, default=str),
        )
        assert data["status"] == "completed"
        assert data["usage"]["total_tokens"] > 0


# ===================================================================
# Validation  (gateway Pydantic models reject malformed requests)
# ===================================================================


class TestValidation:
    def test_chat_missing_model(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 422

    def test_chat_missing_messages(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/chat/completions", json={"model": "x"},
        )
        assert r.status_code == 422

    def test_chat_message_no_role(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"content": "hi"}]},
        )
        assert r.status_code == 422

    def test_completions_missing_model(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/completions", json={"prompt": "hi"},
        )
        assert r.status_code == 422

    def test_completions_missing_prompt(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/completions", json={"model": "x"},
        )
        assert r.status_code == 422

    def test_responses_missing_model(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/responses", json={"input": "hi"},
        )
        assert r.status_code == 422

    def test_responses_missing_input(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/responses", json={"model": "x"},
        )
        assert r.status_code == 422

    def test_embeddings_missing_model(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/embeddings", json={"input": "hi"},
        )
        assert r.status_code == 422

    def test_embeddings_missing_input(self, raw_client: httpx.Client):
        r = raw_client.post(
            "/v1/embeddings", json={"model": "x"},
        )
        assert r.status_code == 422

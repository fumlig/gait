"""Integration tests for the trave gateway.

Requires a running gateway (docker compose up -d). Uses the OpenAI Python
client. Outputs saved to ./test_outputs/ for manual inspection.

    GATEWAY_URL defaults to http://localhost:3000
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
from pathlib import Path

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


class TestHealth:
    def test_health_endpoint(self, raw_client: httpx.Client):
        r = raw_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        _save("health.json", json.dumps(data, indent=2))

        assert "status" in data
        assert "backends" in data
        assert isinstance(data["backends"], dict)
        # At minimum speech + transcription should be present
        assert "speech" in data["backends"]
        assert "transcription" in data["backends"]


class TestModels:
    def test_list_models(self, client: OpenAI, models: dict):
        _save("models.json", json.dumps(models, indent=2))
        assert len(models["all"]) > 0, "No models discovered"

    def test_has_tts_model(self, models: dict):
        assert len(models["tts"]) > 0, "No TTS model found"

    def test_has_stt_model(self, models: dict):
        assert len(models["stt"]) > 0, "No STT model found"

    def test_models_have_capabilities(self, raw_client: httpx.Client):
        """Every model has a non-empty capabilities list."""
        r = raw_client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        _save("models_capabilities.json", json.dumps(data, indent=2))
        for m in data["data"]:
            assert "capabilities" in m, f"Model {m['id']} missing capabilities"
            assert len(m["capabilities"]) > 0, f"Model {m['id']} has empty capabilities"
            assert "loaded" in m, f"Model {m['id']} missing loaded field"


class TestSpeech:
    def test_speech_mp3(self, client: OpenAI, models: dict):
        """Generate speech as MP3 (default format)."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="Hello! This is an integration test of the text to speech system.",
            response_format="mp3",
        )
        audio = resp.read()
        path = _save("speech_mp3.mp3", audio, binary=True)
        assert len(audio) > 100, "MP3 too small"
        assert path.stat().st_size > 100

    def test_speech_wav(self, client: OpenAI, models: dict):
        """Generate speech as WAV."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="Testing WAV output format.",
            response_format="wav",
        )
        audio = resp.read()
        path = _save("speech_wav.wav", audio, binary=True)
        assert audio[:4] == b"RIFF", "Not a valid WAV file"
        assert audio[8:12] == b"WAVE"
        assert path.stat().st_size > 44  # at least the header

    def test_speech_extended_params(self, client: OpenAI, models: dict, raw_client: httpx.Client):
        """Test extended chatterbox parameters via raw HTTP."""
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
        assert r.status_code == 200, f"Extended params failed: {r.text}"
        _save("speech_extended.wav", r.content, binary=True)
        assert r.content[:4] == b"RIFF"

    def test_speech_missing_fields(self, raw_client: httpx.Client):
        """Missing required fields returns 422."""
        r = raw_client.post("/v1/audio/speech", json={"model": "x"})
        assert r.status_code == 422

    def test_speech_unsupported_format(self, client: OpenAI, models: dict, raw_client: httpx.Client):
        """Unsupported format returns 400."""
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


class TestTranscription:
    """Test transcription using TTS-generated audio for a realistic round-trip."""

    @pytest.fixture(scope="class")
    def tts_audio_wav(self, client: OpenAI, models: dict) -> bytes:
        """Generate a WAV via TTS so we have real speech to transcribe."""
        tts_model = models["tts"][0]
        resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input="The quick brown fox jumps over the lazy dog.",
            response_format="wav",
        )
        audio = resp.read()
        _save("tts_for_transcription.wav", audio, binary=True)
        return audio

    def test_transcription_json(self, client: OpenAI, models: dict, tts_audio_wav: bytes):
        """Transcribe with default JSON format."""
        stt_model = models["stt"][0]
        transcript = client.audio.transcriptions.create(
            model=stt_model,
            file=("speech.wav", tts_audio_wav, "audio/wav"),
            response_format="json",
        )
        text = transcript.text
        _save("transcription_json.json", json.dumps({"text": text}, indent=2))
        assert len(text) > 0, "Empty transcription"

    def test_transcription_text(self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client):
        """Transcribe with plain text format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "text"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("transcription_text.txt", r.text)
        assert len(r.text.strip()) > 0

    def test_transcription_verbose_json(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client
    ):
        """Transcribe with verbose_json format — includes segments and words."""
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
        assert data["duration"] > 0
        assert len(data["segments"]) > 0
        assert len(data["text"]) > 0
        # Segments have start/end
        seg = data["segments"][0]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg

    def test_transcription_srt(self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client):
        """Transcribe with SRT subtitle format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "srt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("transcription.srt", r.text)
        assert "-->" in r.text
        assert "1\n" in r.text

    def test_transcription_vtt(self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client):
        """Transcribe with WebVTT subtitle format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "vtt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("transcription.vtt", r.text)
        assert r.text.startswith("WEBVTT")
        assert "-->" in r.text

    def test_transcription_diarize(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client
    ):
        """Transcribe with diarization enabled."""
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
        assert len(data["segments"]) > 0
        assert len(data["text"]) > 0
        # Word-level timestamps should be present (diarization forces them)
        assert len(data["words"]) > 0

    def test_transcription_with_language(
        self, client: OpenAI, models: dict, tts_audio_wav: bytes
    ):
        """Transcribe with explicit language hint."""
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
        assert len(transcript.text) > 0

    def test_transcription_empty_file(self, models: dict, raw_client: httpx.Client):
        """Empty audio file returns 400."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model},
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert r.status_code == 400

    def test_transcription_invalid_format(self, models: dict, raw_client: httpx.Client):
        """Invalid response_format returns 400."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/transcriptions",
            data={"model": stt_model, "response_format": "invalid"},
            files={"file": ("test.wav", _make_wav(), "audio/wav")},
        )
        assert r.status_code == 400


class TestTranslation:
    @pytest.fixture(scope="class")
    def tts_audio_wav(self, client: OpenAI, models: dict) -> bytes:
        """Generate a WAV to use for translation tests."""
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

    def test_translation_json(self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client):
        """Translation returns JSON with English text."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/translations",
            data={"model": stt_model},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        _save("translation_json.json", json.dumps(data, indent=2))
        assert "text" in data
        assert len(data["text"]) > 0

    def test_translation_verbose_json(
        self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client
    ):
        """Translation with verbose_json includes segments."""
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
        assert len(data["segments"]) > 0

    def test_translation_srt(self, models: dict, tts_audio_wav: bytes, raw_client: httpx.Client):
        """Translation with SRT format."""
        stt_model = models["stt"][0]
        r = raw_client.post(
            "/v1/audio/translations",
            data={"model": stt_model, "response_format": "srt"},
            files={"file": ("speech.wav", tts_audio_wav, "audio/wav")},
        )
        assert r.status_code == 200
        _save("translation.srt", r.text)
        assert "-->" in r.text


class TestVoices:
    VOICE_NAME = "integration_test_voice"

    def test_01_list_voices_has_default(self, raw_client: httpx.Client):
        """Voice list always includes the built-in 'default' voice."""
        r = raw_client.get("/v1/audio/voices")
        assert r.status_code == 200
        data = r.json()
        _save("voices_list_initial.json", json.dumps(data, indent=2))
        assert data["object"] == "list"
        names = [v["name"] for v in data["data"]]
        assert "default" in names

    def test_02_get_default_voice(self, raw_client: httpx.Client):
        """Get the default voice by name."""
        r = raw_client.get("/v1/audio/voices/default")
        assert r.status_code == 200
        data = r.json()
        _save("voice_default.json", json.dumps(data, indent=2))
        assert data["voice_id"] == "default"

    def test_03_create_voice(self, raw_client: httpx.Client):
        """Upload a new voice reference clip."""
        wav_data = _make_wav(duration_s=2.0)
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": self.VOICE_NAME},
            files={"file": ("ref.wav", wav_data, "audio/wav")},
        )
        assert r.status_code == 201, f"Create voice failed: {r.text}"
        data = r.json()
        _save("voice_created.json", json.dumps(data, indent=2))
        assert data["voice_id"] == self.VOICE_NAME

    def test_04_list_voices_includes_new(self, raw_client: httpx.Client):
        """New voice appears in the listing."""
        r = raw_client.get("/v1/audio/voices")
        assert r.status_code == 200
        names = [v["name"] for v in r.json()["data"]]
        _save("voices_list_after_create.json", json.dumps(r.json(), indent=2))
        assert self.VOICE_NAME in names

    def test_05_get_created_voice(self, raw_client: httpx.Client):
        """Fetch the created voice by name."""
        r = raw_client.get(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 200
        assert r.json()["voice_id"] == self.VOICE_NAME

    def test_06_create_duplicate_rejected(self, raw_client: httpx.Client):
        """Duplicate voice creation returns 409."""
        wav_data = _make_wav()
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": self.VOICE_NAME},
            files={"file": ("ref.wav", wav_data, "audio/wav")},
        )
        assert r.status_code == 409

    def test_07_create_default_rejected(self, raw_client: httpx.Client):
        """Cannot replace the built-in 'default' voice."""
        wav_data = _make_wav()
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "default"},
            files={"file": ("ref.wav", wav_data, "audio/wav")},
        )
        assert r.status_code == 400

    def test_08_create_invalid_name(self, raw_client: httpx.Client):
        """Invalid voice name returns 400."""
        wav_data = _make_wav()
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "bad name!"},
            files={"file": ("ref.wav", wav_data, "audio/wav")},
        )
        assert r.status_code == 400

    def test_09_create_not_wav(self, raw_client: httpx.Client):
        """Non-WAV file upload returns 400."""
        r = raw_client.post(
            "/v1/audio/voices",
            data={"name": "notwav"},
            files={"file": ("ref.mp3", b"\xff\xfb\x90\x00" + b"\x00" * 100, "audio/mpeg")},
        )
        assert r.status_code == 400

    def test_10_use_custom_voice_for_tts(
        self, client: OpenAI, models: dict, raw_client: httpx.Client
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
        """Delete the test voice."""
        r = raw_client.delete(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 200
        data = r.json()
        _save("voice_deleted.json", json.dumps(data, indent=2))
        assert data["deleted"] is True

    def test_12_delete_not_found(self, raw_client: httpx.Client):
        """Deleting a non-existent voice returns 404."""
        r = raw_client.delete(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 404

    def test_13_delete_default_rejected(self, raw_client: httpx.Client):
        """Cannot delete the built-in 'default' voice."""
        r = raw_client.delete("/v1/audio/voices/default")
        assert r.status_code == 400

    def test_14_get_not_found(self, raw_client: httpx.Client):
        """Fetching a deleted voice returns 404."""
        r = raw_client.get(f"/v1/audio/voices/{self.VOICE_NAME}")
        assert r.status_code == 404


class TestChatCompletions:
    @staticmethod
    def _get_message_text(message) -> str:
        """Return content, falling back to reasoning_content for reasoning models."""
        return message.content or getattr(message, "reasoning_content", "") or ""

    def test_chat_completion(self, client: OpenAI, models: dict):
        """Non-streaming chat completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
            max_tokens=512,
        )
        _save(
            "chat_completion.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        text = self._get_message_text(resp.choices[0].message)
        assert len(text) > 0, "No content or reasoning_content in response"
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0

    def test_chat_completion_stream(self, client: OpenAI, models: dict):
        """Streaming chat completion collects all chunks."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        stream = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=512,
            stream=True,
        )
        chunks = []
        full_text = ""
        full_reasoning = ""
        for chunk in stream:
            chunks.append(chunk.model_dump())
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
            # Reasoning models stream reasoning_content before content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                full_reasoning += reasoning

        _save("chat_stream_chunks.json", json.dumps(chunks, indent=2, default=str))
        _save("chat_stream_text.txt", full_text or full_reasoning)
        assert len(chunks) > 1, "Expected multiple stream chunks"
        assert len(full_text) > 0 or len(full_reasoning) > 0, (
            "No text content or reasoning_content in stream"
        )

    def test_chat_completion_system_message(self, client: OpenAI, models: dict):
        """Chat with system message."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful pirate. Always say 'Arrr!'."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=512,
        )
        _save(
            "chat_system_msg.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        text = self._get_message_text(resp.choices[0].message)
        assert len(text) > 0, "No content or reasoning_content in response"

    def test_chat_completion_multi_turn(self, client: OpenAI, models: dict):
        """Multi-turn conversation."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you?"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=512,
        )
        _save(
            "chat_multi_turn.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        text = self._get_message_text(resp.choices[0].message)
        assert len(text) > 0, "No content or reasoning_content in response"


class TestCompletions:
    def test_text_completion(self, client: OpenAI, models: dict):
        """Non-streaming legacy text completion."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.completions.create(
            model=llm_model,
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
        llm_model = models["llm"][0]
        stream = client.completions.create(
            model=llm_model,
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

        _save("completion_stream_chunks.json", json.dumps(chunks, indent=2, default=str))
        _save("completion_stream_text.txt", full_text)
        assert len(chunks) > 1
        assert len(full_text) > 0


class TestResponses:
    def test_response_non_streaming(self, models: dict, raw_client: httpx.Client):
        """Non-streaming response returns structured output."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": llm_model,
                "input": "Say hello in one word.",
                "max_output_tokens": 512,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save("response_non_streaming.json", json.dumps(data, indent=2, default=str))

        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert len(data["output"]) > 0, "No output items"
        assert data["usage"]["total_tokens"] > 0

        # Should contain at least reasoning or message output
        output_types = [item["type"] for item in data["output"]]
        assert any(t in ("reasoning", "message") for t in output_types), (
            f"Expected reasoning or message output, got: {output_types}"
        )

    def test_response_streaming(self, models: dict, raw_client: httpx.Client):
        """Streaming response returns SSE events."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": llm_model,
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
                event_type = line[7:]
                events.append(event_type)

        _save("response_streaming_events.json", json.dumps(events, indent=2))

        # Must have the core lifecycle events
        assert "response.created" in events, "Missing response.created event"
        assert "response.completed" in events or "response.done" in events, (
            "Missing response completion event"
        )
        # Should have some content deltas (reasoning or output text)
        has_content = any(
            "delta" in e or "text" in e for e in events
        )
        assert has_content, f"No content delta events found in: {events}"

    def test_response_with_instructions(self, models: dict, raw_client: httpx.Client):
        """Response with system-level instructions."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        r = raw_client.post(
            "/v1/responses",
            json={
                "model": llm_model,
                "instructions": "You are a pirate. Always say Arrr!",
                "input": "Hello!",
                "max_output_tokens": 512,
            },
        )
        assert r.status_code == 200
        data = r.json()
        _save("response_with_instructions.json", json.dumps(data, indent=2, default=str))
        assert data["status"] == "completed"
        assert len(data["output"]) > 0


class TestEmbeddings:
    def test_single_embedding(self, client: OpenAI, models: dict):
        """Get embedding for a single string."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.embeddings.create(
            model=llm_model,
            input="Hello world",
        )
        _save(
            "embedding_single.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(resp.data) == 1
        assert len(resp.data[0].embedding) > 0, "Empty embedding vector"

    def test_batch_embeddings(self, client: OpenAI, models: dict):
        """Get embeddings for multiple inputs."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]
        resp = client.embeddings.create(
            model=llm_model,
            input=["Hello", "World", "Test"],
        )
        _save(
            "embedding_batch.json",
            json.dumps(resp.model_dump(), indent=2, default=str),
        )
        assert len(resp.data) == 3
        for item in resp.data:
            assert len(item.embedding) > 0


class TestRoundTrip:
    """Generate speech and then transcribe it back — the ultimate sanity check."""

    def test_tts_to_stt_round_trip(self, client: OpenAI, models: dict):
        """Speak text, transcribe it, check the round-trip makes sense."""
        if not models["tts"] or not models["stt"]:
            pytest.skip("Need both TTS and STT models")

        original_text = "The quick brown fox jumps over the lazy dog."
        tts_model = models["tts"][0]
        stt_model = models["stt"][0]

        # TTS
        tts_resp = client.audio.speech.create(
            model=tts_model,
            voice="default",
            input=original_text,
            response_format="wav",
        )
        audio = tts_resp.read()
        _save("roundtrip_tts.wav", audio, binary=True)

        # STT
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
        assert len(transcript.text) > 0, "Empty transcription in round-trip"
        # Check some words survived the round trip
        original_words = set(original_text.lower().split())
        transcribed_words = set(transcript.text.lower().replace(".", "").replace(",", "").split())
        overlap = original_words & transcribed_words
        assert len(overlap) >= 3, (
            f"Round-trip lost too many words. "
            f"Original: {original_words}, Got: {transcribed_words}, Overlap: {overlap}"
        )


class TestChatAudio:
    def test_chat_audio_stream(self, models: dict, raw_client: httpx.Client):
        """Streaming with audio modality interleaves text and audio SSE events."""
        if not models["llm"] or not models["tts"]:
            pytest.skip("Need both LLM and TTS models")

        llm_model = models["llm"][0]
        tts_model = models["tts"][0]

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 512,
                "stream": True,
                "modalities": ["text", "audio"],
                "audio": {"voice": "default", "model": tts_model},
            },
            # Don't follow the streaming auto-read; read raw
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

        raw_sse = r.text
        _save("chat_audio_stream.txt", raw_sse)

        text_tokens = []
        reasoning_tokens = []
        audio_data_chunks = []
        audio_transcripts = []
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
            if "content" in delta and delta["content"]:
                text_tokens.append(delta["content"])
            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning_tokens.append(delta["reasoning_content"])
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
                    "text_tokens": text_tokens,
                    "reasoning_tokens_count": len(reasoning_tokens),
                    "audio_chunks_count": len(audio_data_chunks),
                    "audio_transcripts": audio_transcripts,
                    "got_audio_id": got_audio_id,
                    "got_expires": got_expires,
                },
                indent=2,
            ),
        )

        # Either content or reasoning_content must be present
        assert len(text_tokens) > 0 or len(reasoning_tokens) > 0, (
            "No text tokens or reasoning tokens in audio stream"
        )
        assert got_audio_id, "Missing audio.id in stream"
        assert got_expires, "Missing audio.expires_at in stream"
        # Audio data + transcripts should be present when content was generated
        # For reasoning models, TTS only fires on content tokens, so audio
        # may be absent if the model used all tokens on reasoning.
        if len(text_tokens) > 0:
            assert len(audio_data_chunks) > 0, "No audio data chunks despite content tokens"
            assert len(audio_transcripts) > 0, "No audio transcripts despite content tokens"

    def test_chat_audio_requires_stream(self, models: dict, raw_client: httpx.Client):
        """Audio modality without stream=true returns 400."""
        if not models["llm"]:
            pytest.skip("No LLM model available")
        llm_model = models["llm"][0]

        r = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": "Hi"}],
                "modalities": ["text", "audio"],
                "audio": {"voice": "default"},
            },
        )
        assert r.status_code == 400

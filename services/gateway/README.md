# gateway

FastAPI gateway that exposes OpenAI-compatible REST APIs for all trave backend services. Handles request validation, response formatting (WAV→MP3, segments→SRT/VTT), and model list aggregation.

## Endpoints

### Chat

#### `POST /v1/chat/completions`

Create a chat completion. Transparently proxied to the llama.cpp server backend. Supports streaming (`"stream": true`). See [llamacpp README](../llamacpp/README.md) for model configuration.

### Completions

#### `POST /v1/completions`

Create a text completion (legacy). Transparently proxied to the llama.cpp server backend. Supports streaming (`"stream": true`).

### Responses

#### `POST /v1/responses`

Create a model response (OpenAI Responses API). Transparently proxied to the llama.cpp server backend. Supports streaming (`"stream": true`). See [OpenAI Responses API docs](https://platform.openai.com/docs/api-reference/responses).

### Embeddings

#### `POST /v1/embeddings`

Create embeddings for the given input. Transparently proxied to the llama.cpp server backend.

### Audio

#### `POST /v1/audio/speech`

Generate speech. Maps the OpenAI TTS API to the chatterbox backend's `/synthesize` endpoint. See [chatterbox README](../chatterbox/README.md) for model-specific parameters.

#### `POST /v1/audio/transcriptions`

Transcribe audio. Maps the OpenAI Transcription API to the whisperx backend's `/transcribe` endpoint. See [whisperx README](../whisperx/README.md) for model details.

Supports `response_format`: `json`, `text`, `srt`, `verbose_json`, `vtt`. Format conversion from raw backend JSON is handled by the gateway.

#### `POST /v1/audio/translations`

Translate audio to English. Maps to whisperx's `/translate` endpoint. Same format options as transcriptions.

#### Voice management

- `GET /v1/audio/voices` — list voices
- `GET /v1/audio/voices/{name}` — get a voice
- `POST /v1/audio/voices` — create a voice (multipart)
- `DELETE /v1/audio/voices/{name}` — delete a voice

Managed locally on the gateway's filesystem. The `VOICES_DIR` directory is shared with the chatterbox container via a Docker volume so uploaded voice clips are immediately available for TTS inference. A virtual `default` voice is always present (maps to chatterbox's built-in reference clip).

### Models

#### `GET /v1/models`

Returns a merged list of models from all backend services. Models are fetched once at startup and cached.

### Health

#### `GET /health`

Returns gateway status and per-backend health:

```json
{
  "status": "ok",
  "backends": {
    "llamacpp": "healthy",
    "chatterbox": "healthy",
    "whisperx": "healthy"
  }
}
```

Status is `"ok"` when all backends are healthy, `"degraded"` otherwise. Only backends whose env var is set appear in the response.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `LLAMACPP_URL` | _(unset)_ | URL for the llama.cpp (LLM) backend. Unset = disabled |
| `CHATTERBOX_URL` | _(unset)_ | URL for the chatterbox (TTS) backend. Unset = disabled |
| `WHISPERX_URL` | _(unset)_ | URL for the whisperx (STT) backend. Unset = disabled |
| `VOICES_DIR` | _(unset)_ | Local directory for voice WAV clips (shared volume with chatterbox). Unset = voice management disabled |
| `BACKEND_TIMEOUT` | `300` | Timeout in seconds for backend requests |

Each backend is optional — set its URL env var to enable it, or leave it unset to disable. The gateway starts with whatever backends are configured.

## Architecture

```
src/gateway_service/
  main.py          # FastAPI app + lifespan (backend discovery, model fetch)
  config.py        # Pydantic settings (host, port, timeout)
  models.py        # Response schemas (ModelObject, HealthResponse, etc.)
  formatting.py    # Format conversion (WAV→MP3, segments→SRT/VTT)
  protocols.py     # Resource protocols (ChatCompletions, AudioSpeech, etc.)
  backends/
    base.py          # BaseBackend base class (health, models, create)
    __init__.py      # KNOWN_BACKENDS registry, Registerable protocol
    llamacpp.py      # LlamacppClient — transparent proxy for llama.cpp
    chatterbox.py    # ChatterboxClient — TTS via chatterbox
    whisperx.py      # WhisperxClient — STT via whisperx
    voice.py         # VoiceClient — local filesystem voice management
  routes/
    health.py        # GET /health (aggregated backend status)
    models.py        # GET /v1/models (cached model list)
    completions.py   # POST /v1/completions
    responses.py     # POST /v1/responses
    embeddings.py    # POST /v1/embeddings
    chat/
      completions.py   # POST /v1/chat/completions
    audio/
      speech.py          # POST /v1/audio/speech
      transcriptions.py  # POST /v1/audio/transcriptions
      translations.py    # POST /v1/audio/translations
      voices.py          # Voice CRUD
```

### Backend client system

Each OpenAI resource group has a **protocol** defined in `protocols.py` (e.g. `ChatCompletions`, `AudioSpeech`, `AudioVoices`). Backend clients implement the protocols for the resources they support. At startup the gateway uses `isinstance` checks to automatically wire each client to the correct `app.state` slot.

Each client class:

1. **Extends `BaseBackend`** (for HTTP backends) or implements resource protocols directly (for local backends like `VoiceClient`).
2. **Inherits its resource protocols** — e.g. `class ChatterboxClient(BaseBackend, AudioSpeech)`. This gives static type checking that all protocol methods are implemented correctly.
3. **Declares `env_var`** and a **`create` classmethod** — the gateway calls `cls.create(env_value, http_client)` to instantiate uniformly.
4. **Is registered** in `KNOWN_BACKENDS` in `backends/__init__.py`.

At startup, the gateway iterates `KNOWN_BACKENDS`, checks `os.environ` for each class's `env_var`, and calls `create` to instantiate only those whose variable is set. It then uses `isinstance` against each protocol in `PROTOCOL_SLOTS` to wire `app.state` slots automatically.

### Adding a new backend client

To connect a new backend service to the gateway:

**1. Create the client class** in `backends/<name>.py`:

```python
from typing import ClassVar
from gateway_service.backends.base import BaseBackend
from gateway_service.protocols import AudioSpeech, AudioTranscriptions, AudioTranslations

class MyServiceClient(BaseBackend, AudioSpeech, AudioTranscriptions, AudioTranslations):
    name = "myservice"
    env_var = "MYSERVICE_URL"
    default_model_capabilities: ClassVar[list[str]] = ["speech", "transcription"]

    async def synthesize(self, request) -> tuple[bytes, str]:
        """TTS — matches the AudioSpeech protocol."""
        ...

    async def transcribe(self, *, file, filename, model, ...) -> TranscriptionResult:
        """STT — matches the AudioTranscriptions protocol."""
        ...

    async def translate(self, *, file, filename, model, ...) -> TranscriptionResult:
        """Translation — matches the AudioTranslations protocol."""
        ...
```

The protocols the class inherits determine which `app.state` slots it fills. Inheriting `AudioSpeech` and `AudioTranscriptions` means this single backend handles both TTS and STT routes — replacing both `ChatterboxClient` and `WhisperxClient`. The static type checker (`ty`) verifies all protocol methods are implemented correctly.

**2. Register it** in `backends/__init__.py`:

```python
from gateway_service.backends.myservice import MyServiceClient

KNOWN_BACKENDS: list[type[Registerable]] = [
    LlamacppClient,
    MyServiceClient,  # replaces ChatterboxClient + WhisperxClient
    VoiceClient,
]
```

**3. Set the env var** in `docker-compose.yml`:

```yaml
gateway:
  environment:
    - LLAMACPP_URL=http://llamacpp:8000
    - MYSERVICE_URL=http://myservice:8000
    - VOICES_DIR=/app/voices
```

That's it. The gateway will instantiate `MyServiceClient`, discover via `isinstance` that it implements `AudioSpeech`, `AudioTranscriptions`, and `AudioTranslations`, wire the matching `app.state` slots, fetch models from it, and include it in health checks — all automatically.

## Development

Requires Python 3.12 (pinned in `.python-version`; `uv` auto-downloads it if missing).

```bash
uv sync --all-extras      # creates .venv with Python 3.12, installs all deps + dev tools
uv run ruff check src/    # lint
uvx ty check              # type check
uv run pytest             # tests (mocked clients, no backends needed)
```

For IDE support (e.g. VS Code / PyCharm), point the Python interpreter at `services/gateway/.venv/bin/python`.

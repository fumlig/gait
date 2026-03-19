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
    "speech": "healthy",
    "transcription": "healthy",
    "chat": "healthy"
  }
}
```

Status is `"ok"` when all backends are healthy, `"degraded"` otherwise.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `SPEECH_URL` | `http://chatterbox:8000` | Internal URL for the chatterbox (TTS) service |
| `TRANSCRIPTION_URL` | `http://whisperx:8000` | Internal URL for the whisperx (STT) service |
| `CHAT_URL` | _(empty)_ | Internal URL for the llama.cpp (LLM) service. Empty = disabled |
| `VOICES_DIR` | `/app/voices` | Local directory for voice reference WAV clips (shared volume with chatterbox) |
| `BACKEND_TIMEOUT` | `300` | Timeout in seconds for backend requests |

## Development

Requires Python 3.12 (pinned in `.python-version`; `uv` auto-downloads it if missing).

```bash
uv sync --all-extras      # creates .venv with Python 3.12, installs all deps + dev tools
uv run ruff check src/    # lint
uv run pytest             # tests (mocked clients, no backends needed)
```

For IDE support (e.g. VS Code / PyCharm), point the Python interpreter at `services/gateway/.venv/bin/python`.

## Architecture

```
src/gateway_service/
  main.py          # FastAPI app + lifespan (client init, model discovery)
  config.py        # Pydantic settings from env vars
  models.py        # Response schemas (ModelObject, HealthResponse, etc.)
  formatting.py    # Format conversion (WAV→MP3, segments→SRT/VTT)
  clients/
    chat.py          # Transparent proxy client for llama.cpp server
    speech.py        # Typed HTTP client for chatterbox (/synthesize)
    transcription.py # Typed HTTP client for whisperx (/transcribe, /translate)
    voice.py         # Local filesystem client for voice management
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
      voices.py          # Voice management (proxied to voice service)
```

Clients are stored on `app.state` (e.g. `request.app.state.speech_client`). No registry or dynamic discovery — backend URLs come from static environment variables.

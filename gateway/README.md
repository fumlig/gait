# gateway

FastAPI gateway that exposes OpenAI-compatible REST APIs for all gait backends. Handles request validation, response formatting (WAV→MP3, segments→SRT/VTT), and model list aggregation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported, audio interleaving) |
| `POST` | `/v1/completions` | Text completions (streaming supported) |
| `POST` | `/v1/responses` | Responses API (streaming supported) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/audio/speech` | Text-to-speech (WAV, MP3) |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (json, text, srt, vtt, verbose_json) |
| `POST` | `/v1/audio/translations` | Translate audio to English (same formats as transcriptions) |
| `GET` | `/v1/audio/voices` | List voices |
| `POST` | `/v1/audio/voices` | Create voice (multipart: name + WAV file) |
| `GET` | `/v1/audio/voices/{name}` | Get voice |
| `DELETE` | `/v1/audio/voices/{name}` | Delete voice |
| `GET` | `/v1/models` | Merged model list from all backends (auto-refreshing) |
| `GET` | `/health` | Gateway + per-backend health status |

LLM endpoints are proxied transparently to llama.cpp. Audio endpoints convert between OpenAI API formats and backend RPC formats.

Voice clips are stored on a local directory shared with chatterbox via Docker volume. A virtual `default` voice is always present (maps to chatterbox's built-in reference clip).

When `/v1/chat/completions` includes `modalities: ["text", "audio"]` with `stream: true`, the gateway interleaves text deltas with PCM16 audio chunks synthesised by the TTS backend.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `LLAMACPP_URL` | _(unset)_ | llama.cpp backend URL (unset = disabled) |
| `CHATTERBOX_URL` | _(unset)_ | Chatterbox backend URL (unset = disabled) |
| `WHISPERX_URL` | _(unset)_ | WhisperX backend URL (unset = disabled) |
| `VOICES_DIR` | _(unset)_ | Voice clips directory (unset = voice management disabled) |
| `BACKEND_TIMEOUT` | `300` | Backend request timeout (seconds) |

Each backend is optional — set its URL to enable it.

## Provider client system

Each OpenAI resource group has a protocol in `protocols.py`. Provider clients implement the protocols they support. At startup the gateway iterates `KNOWN_PROVIDERS`, instantiates each client whose env var is set, and uses `isinstance` checks to wire `app.state` slots automatically.

To add a new provider: create a client class in `providers/`, extend `BaseProvider`, implement the relevant protocols, register it in `KNOWN_PROVIDERS`.

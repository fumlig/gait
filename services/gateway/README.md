# gateway

FastAPI reverse proxy that unifies all trave services behind a single port. Routes requests to the appropriate backend service based on the API path, and merges `/v1/models` responses from all backends.

## Endpoints

### `POST /v1/audio/speech`

Proxied to the **chatterbox** backend. See [chatterbox README](../chatterbox/README.md) for parameters.

### `POST /v1/audio/transcriptions`

Proxied to the **whisperx** backend. See [whisperx README](../whisperx/README.md) for parameters.

### `POST /v1/audio/translations`

Proxied to the **whisperx** backend. See [whisperx README](../whisperx/README.md) for parameters.

### `GET /v1/models`

Returns a merged, deduplicated list of models from all backend services. If a backend is unreachable, its models are omitted (no error).

### `GET /health`

Returns gateway status and per-backend health:

```json
{
  "status": "ok",
  "backends": {
    "chatterbox": "healthy",
    "whisperx": "healthy"
  }
}
```

Status is `"ok"` when all backends are healthy, `"degraded"` otherwise.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `CHATTERBOX_URL` | `http://chatterbox:8000` | Internal URL for the chatterbox service |
| `WHISPERX_URL` | `http://whisperx:8000` | Internal URL for the whisperx service |
| `PROXY_TIMEOUT` | `300` | Timeout in seconds for proxied requests |

## Development

```bash
uv sync --dev
uv run ruff check src/
uv run pytest
```

Tests mock the httpx client and do not require running backend services.

## Architecture

```
src/gateway_service/
  main.py          # FastAPI app + lifespan (httpx client lifecycle)
  config.py        # Pydantic settings from env vars
  models.py        # Response schemas
  proxy.py         # httpx reverse proxy logic
  routes/
    audio.py       # POST /v1/audio/* (proxied)
    models.py      # GET /v1/models (merged from backends)
    health.py      # GET /health (aggregated backend status)
```

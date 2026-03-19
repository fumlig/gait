# voice

Lightweight voice management service for WAV reference clips. Runs as a Starlette app with no GPU or ML dependencies. Shares a `VOICES_DIR` volume with the chatterbox service so that uploaded voice clips are immediately available for TTS inference.

## Endpoints

### `GET /voices`

List all registered voices.

```json
{
  "object": "list",
  "data": [
    {"voice_id": "narrator", "name": "narrator"},
    {"voice_id": "emma", "name": "emma"}
  ]
}
```

### `GET /voices/{name}`

Get a single voice by name. Returns 404 if not found.

### `POST /voices`

Upload a new voice reference clip.

**Multipart form data:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | yes | Voice name (alphanumeric, hyphens, underscores) |
| `file` | file | yes | WAV audio file (~10 seconds of clean speech) |

Returns 201 on success, 409 if the voice already exists, 400 for invalid input.

### `DELETE /voices/{name}`

Delete a voice reference clip. Returns 404 if not found.

### `GET /health`

Returns `{"status": "ok"}`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `VOICES_DIR` | `/app/voices` | Path to voice reference clips inside the container |

## Development

Requires Python 3.12 (pinned in `.python-version`; `uv` auto-downloads it if missing).

```bash
uv sync --all-extras      # creates .venv with Python 3.12, installs all deps + dev tools
uv run ruff check src/    # lint
uv run pytest             # tests
```

For IDE support (e.g. VS Code / PyCharm), point the Python interpreter at `services/voice/.venv/bin/python`.

## Build notes

Uses `python:3.12-slim` base image (no GPU required). Only dependencies are Starlette, uvicorn, and pydantic-settings.

## Architecture

```
src/voice_service/
  app.py         # Starlette app (CRUD endpoints: /voices, /health)
  config.py      # Pydantic settings from env vars
```

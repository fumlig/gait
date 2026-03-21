from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from gateway_service.deps import (
    VoicesClient,  # noqa: TC001 — FastAPI resolves Annotated at runtime
)

router = APIRouter()


@router.get("/v1/audio/voices")
async def list_voices(client: VoicesClient) -> dict[str, object]:
    voices = await client.list_voices()
    return {"object": "list", "data": [v.model_dump() for v in voices]}


@router.get("/v1/audio/voices/{name}")
async def get_voice(name: str, client: VoicesClient) -> dict[str, str]:
    voice = await client.get_voice(name)
    return voice.model_dump()


@router.post("/v1/audio/voices", status_code=201)
async def create_voice(
    client: VoicesClient,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, str]:
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    voice = await client.create_voice(name, audio_data)
    return voice.model_dump()


@router.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str, client: VoicesClient) -> dict[str, object]:
    return await client.delete_voice(name)

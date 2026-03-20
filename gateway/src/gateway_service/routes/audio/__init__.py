from fastapi import APIRouter

from gateway_service.routes.audio.speech import router as speech_router
from gateway_service.routes.audio.transcriptions import router as transcriptions_router
from gateway_service.routes.audio.translations import router as translations_router
from gateway_service.routes.audio.voices import router as voices_router

router = APIRouter()
router.include_router(speech_router)
router.include_router(transcriptions_router)
router.include_router(translations_router)
router.include_router(voices_router)

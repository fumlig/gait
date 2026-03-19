"""Chat route group — POST /v1/chat/completions."""

from gateway_service.routes.chat.completions import router

__all__ = ["router"]

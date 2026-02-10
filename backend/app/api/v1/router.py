from fastapi import APIRouter

from app.api.v1 import health, documents, classification, search, chat, agents, auth

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(classification.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(agents.router)

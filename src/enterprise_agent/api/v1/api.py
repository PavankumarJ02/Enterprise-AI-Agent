"""Aggregate API v1 routes."""

from fastapi import APIRouter

from enterprise_agent.api.v1.chat import router as chat_router
from enterprise_agent.api.v1.health import router as health_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(chat_router)

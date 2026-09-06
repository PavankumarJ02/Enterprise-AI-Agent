"""Aggregate API v1 routes."""

from fastapi import APIRouter

from enterprise_agent.api.v1.agent import router as agent_router
from enterprise_agent.api.v1.chat import router as chat_router
from enterprise_agent.api.v1.documents import router as documents_router
from enterprise_agent.api.v1.embeddings import router as embeddings_router
from enterprise_agent.api.v1.evaluation import router as evaluation_router
from enterprise_agent.api.v1.experiments import router as experiments_router
from enterprise_agent.api.v1.guardrails import router as guardrails_router
from enterprise_agent.api.v1.health import router as health_router
from enterprise_agent.api.v1.observability import router as observability_router
from enterprise_agent.api.v1.rag import router as rag_router
from enterprise_agent.api.v1.router import router as query_router
from enterprise_agent.api.v1.search import router as search_router
from enterprise_agent.api.v1.sql import router as sql_router
from enterprise_agent.api.v1.transformation import router as transform_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(embeddings_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(rag_router)
api_v1_router.include_router(transform_router)
api_v1_router.include_router(agent_router)
api_v1_router.include_router(sql_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(guardrails_router)
api_v1_router.include_router(evaluation_router)
api_v1_router.include_router(experiments_router)
api_v1_router.include_router(observability_router)

"""Unit tests for QueryRouterService cascade orchestration and dispatch."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from enterprise_agent.agent.service import AgentService
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.rag.service import RAGService
from enterprise_agent.router.heuristics import HeuristicRouter
from enterprise_agent.router.llm import LLMRouter
from enterprise_agent.router.semantic import SemanticEmbeddingRouter
from enterprise_agent.router.service import QueryRouterService
from enterprise_agent.schemas.agent import AgentChatResponse
from enterprise_agent.schemas.chat import TokenUsageResponse
from enterprise_agent.schemas.rag import RAGQueryResponse
from enterprise_agent.schemas.router import (
    QueryIntent,
    RouterDispatchRequest,
)
from enterprise_agent.sql.service import SQLDatabaseService


@pytest.fixture
def mock_subsystems() -> tuple[
    HeuristicRouter,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
]:
    heur = HeuristicRouter()
    sem = AsyncMock(spec=SemanticEmbeddingRouter)
    llm_r = AsyncMock(spec=LLMRouter)
    llm = AsyncMock(spec=LLMProvider)
    rag = AsyncMock(spec=RAGService)
    sql = AsyncMock(spec=SQLDatabaseService)
    agent = AsyncMock(spec=AgentService)
    return heur, sem, llm_r, llm, rag, sql, agent


@pytest.mark.asyncio
async def test_cascade_tier1_heuristic_short_circuits(
    mock_subsystems: tuple[Any, ...],
) -> None:
    """Verify greeting hits heuristic tier 1 and bypasses semantic and LLM tiers."""
    heur, sem, llm_r, llm, rag, sql, agent = mock_subsystems
    service = QueryRouterService(
        heuristic_router=heur,
        semantic_router=sem,
        llm_router=llm_r,
        llm=llm,
        rag_service=rag,
        sql_service=sql,
        agent_service=agent,
        strategy="cascade",
    )

    decision = await service.classify("Good morning!")
    assert decision.intent == QueryIntent.DIRECT_CHAT
    assert decision.strategy_used == "heuristic"
    sem.classify.assert_not_called()
    llm_r.classify.assert_not_called()


@pytest.mark.asyncio
async def test_cascade_tier2_semantic_short_circuits(
    mock_subsystems: tuple[Any, ...],
) -> None:
    """Verify heuristic miss falls back to semantic tier and short-circuits LLM."""
    heur, sem, llm_r, llm, rag, sql, agent = mock_subsystems
    sem.classify.return_value = (
        QueryIntent.RAG_SEARCH,
        0.88,
        "Semantic similarity matched policy.",
    )

    service = QueryRouterService(
        heuristic_router=heur,
        semantic_router=sem,
        llm_router=llm_r,
        llm=llm,
        strategy="cascade",
    )

    decision = await service.classify("Explain the corporate patent filing procedures.")
    assert decision.intent == QueryIntent.RAG_SEARCH
    assert decision.strategy_used == "semantic"
    sem.classify.assert_called_once()
    llm_r.classify.assert_not_called()


@pytest.mark.asyncio
async def test_cascade_tier3_llm_fallback(
    mock_subsystems: tuple[Any, ...],
) -> None:
    """Verify when heuristic and semantic return None, LLM tier classifies query."""
    heur, sem, llm_r, llm, rag, sql, agent = mock_subsystems
    sem.classify.return_value = None
    llm_r.classify.return_value = (
        QueryIntent.AUTONOMOUS_AGENT,
        0.91,
        "Complex decision requiring multi-tool synthesis.",
    )

    service = QueryRouterService(
        heuristic_router=heur,
        semantic_router=sem,
        llm_router=llm_r,
        llm=llm,
        strategy="cascade",
    )

    decision = await service.classify("Ambiguous query needing LLM reasoning")
    assert decision.intent == QueryIntent.AUTONOMOUS_AGENT
    assert decision.strategy_used == "llm"
    llm_r.classify.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_executes_rag_service(
    mock_subsystems: tuple[Any, ...],
) -> None:
    """Verify dispatching RAG intent invokes rag_service.query."""
    heur, sem, llm_r, llm, rag, sql, agent = mock_subsystems
    rag.query.return_value = RAGQueryResponse(
        query="What is the company parental leave policy?",
        answer="Parental leave is 16 weeks fully paid.",
        sources=[],
        model="mock-rag",
        usage=TokenUsageResponse(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=12.5,
    )

    service = QueryRouterService(
        heuristic_router=heur,
        semantic_router=sem,
        llm_router=llm_r,
        llm=llm,
        rag_service=rag,
        strategy="cascade",
    )

    req = RouterDispatchRequest(
        query="What is the company parental leave policy?",
        force_intent=QueryIntent.RAG_SEARCH,
    )
    result = await service.dispatch(req)

    assert result.intent == QueryIntent.RAG_SEARCH
    rag.query.assert_called_once()
    assert "16 weeks" in result.response["answer"]


@pytest.mark.asyncio
async def test_dispatch_executes_agent_service(
    mock_subsystems: tuple[Any, ...],
) -> None:
    """Verify dispatching AUTONOMOUS_AGENT intent invokes agent_service.run."""
    heur, sem, llm_r, llm, rag, sql, agent = mock_subsystems
    agent.run.return_value = AgentChatResponse(
        query="Find and calculate",
        final_answer="The total cost is $3,200.",
        steps=[],
        iterations=2,
        total_duration_ms=45.0,
        status="success",
    )

    service = QueryRouterService(
        heuristic_router=heur,
        semantic_router=sem,
        llm_router=llm_r,
        llm=llm,
        agent_service=agent,
        strategy="cascade",
    )

    req = RouterDispatchRequest(query="Look up the policy and calculate total bonus")
    result = await service.dispatch(req)

    assert result.intent == QueryIntent.AUTONOMOUS_AGENT
    agent.run.assert_called_once()
    assert "$3,200" in result.response["final_answer"]

"""Query Router Service orchestrating cascade classification and unified subsystem dispatch."""

import time
from typing import Any

from enterprise_agent.agent.service import AgentService
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.rag.service import RAGService
from enterprise_agent.router.heuristics import HeuristicRouter
from enterprise_agent.router.llm import LLMRouter
from enterprise_agent.router.semantic import SemanticEmbeddingRouter
from enterprise_agent.schemas.agent import AgentChatRequest
from enterprise_agent.schemas.rag import RAGQueryRequest
from enterprise_agent.schemas.router import (
    QueryIntent,
    RouterClassifyResponse,
    RouterDispatchRequest,
    RouterDispatchResponse,
)
from enterprise_agent.sql.service import SQLDatabaseService

logger = get_logger(__name__)


class QueryRouterService:
    """Classifies user queries across 3-tier cascade and dispatches to appropriate subsystem."""

    def __init__(
        self,
        heuristic_router: HeuristicRouter,
        semantic_router: SemanticEmbeddingRouter,
        llm_router: LLMRouter,
        llm: LLMProvider,
        rag_service: RAGService | None = None,
        sql_service: SQLDatabaseService | None = None,
        agent_service: AgentService | None = None,
        strategy: str = "cascade",
    ) -> None:
        self.heuristic_router = heuristic_router
        self.semantic_router = semantic_router
        self.llm_router = llm_router
        self.llm = llm
        self.rag_service = rag_service
        self.sql_service = sql_service
        self.agent_service = agent_service
        self.strategy = strategy.lower()

    async def classify(self, query: str) -> RouterClassifyResponse:
        """Classify user query intent using configured strategy or full cascade."""
        start_time = time.perf_counter()
        text = query.strip()

        # Strategy 1: Explicit Heuristic only
        if self.strategy == "heuristic":
            heur_result = self.heuristic_router.classify(text)
            intent, conf, reason = (
                heur_result
                if heur_result
                else (QueryIntent.DIRECT_CHAT, 0.50, "Default fallback for heuristic router.")
            )
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return RouterClassifyResponse(
                query=text,
                intent=intent,
                confidence=conf,
                strategy_used="heuristic",
                reasoning=reason,
                latency_ms=elapsed,
            )

        # Strategy 2: Explicit Semantic only
        if self.strategy == "semantic":
            sem_result = await self.semantic_router.classify(text)
            intent, conf, reason = (
                sem_result
                if sem_result
                else (QueryIntent.DIRECT_CHAT, 0.50, "Below semantic threshold; default fallback.")
            )
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return RouterClassifyResponse(
                query=text,
                intent=intent,
                confidence=conf,
                strategy_used="semantic",
                reasoning=reason,
                latency_ms=elapsed,
            )

        # Strategy 3: Explicit LLM only
        if self.strategy == "llm":
            intent, conf, reason = await self.llm_router.classify(text)
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return RouterClassifyResponse(
                query=text,
                intent=intent,
                confidence=conf,
                strategy_used="llm",
                reasoning=reason,
                latency_ms=elapsed,
            )

        # Default: 3-Tier Cascade (Heuristic -> Semantic -> LLM)
        # Tier 1: Fast Heuristics (< 1ms)
        heur_match = self.heuristic_router.classify(text)
        if heur_match:
            intent, conf, reason = heur_match
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return RouterClassifyResponse(
                query=text,
                intent=intent,
                confidence=conf,
                strategy_used="heuristic",
                reasoning=reason,
                latency_ms=elapsed,
            )

        # Tier 2: Semantic Embedding Cosine Match (~10-50ms)
        sem_match = await self.semantic_router.classify(text)
        if sem_match:
            intent, conf, reason = sem_match
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return RouterClassifyResponse(
                query=text,
                intent=intent,
                confidence=conf,
                strategy_used="semantic",
                reasoning=reason,
                latency_ms=elapsed,
            )

        # Tier 3: Zero-shot LLM Classification fallback (~200-500ms)
        intent, conf, reason = await self.llm_router.classify(text)
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return RouterClassifyResponse(
            query=text,
            intent=intent,
            confidence=conf,
            strategy_used="llm",
            reasoning=reason,
            latency_ms=elapsed,
        )

    async def dispatch(self, request: RouterDispatchRequest) -> RouterDispatchResponse:
        """Classify query intent and execute the appropriate enterprise answering engine."""
        overall_start = time.perf_counter()
        query_text = request.query.strip()

        # Determine intent (either forced or via classifier)
        if request.force_intent:
            intent = request.force_intent
            confidence = 1.0
            strategy = "forced_override"
            reasoning = f"User explicitly forced intent '{intent.value}'."
        else:
            decision = await self.classify(query_text)
            intent = decision.intent
            confidence = decision.confidence
            strategy = decision.strategy_used
            reasoning = decision.reasoning

        logger.info("Routing query '%s' to %s via %s", query_text, intent.value, strategy)
        payload: Any = None

        # Execute target subsystem
        if intent == QueryIntent.DIRECT_CHAT:
            prompt = (
                "You are an enterprise AI assistant. "
                "Answer the user conversationally and concisely."
            )
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=prompt),
                ChatMessage(role=MessageRole.USER, content=query_text),
            ]
            llm_res = await self.llm.generate(
                messages=messages,
                temperature=request.temperature or 0.2,
                max_tokens=512,
            )
            payload = {
                "answer": llm_res.content,
                "model": llm_res.model,
                "tokens": llm_res.usage.total_tokens,
            }

        elif intent == QueryIntent.RAG_SEARCH:
            if self.rag_service:
                rag_req = RAGQueryRequest(query=query_text, top_k=4)
                rag_resp = await self.rag_service.query(rag_req)
                payload = rag_resp.model_dump()
            else:
                payload = {"error": "RAGService not configured."}

        elif intent == QueryIntent.SQL_DATABASE:
            # If raw SQL query, execute directly
            if query_text.upper().startswith(("SELECT", "WITH")) and self.sql_service:
                sql_resp = self.sql_service.execute_query(query_text)
                payload = sql_resp.model_dump()
            elif self.agent_service:
                # Let agent inspect schema and query SQL for natural language analytics
                agent_req = AgentChatRequest(
                    query=query_text,
                    max_iterations=request.max_iterations or 4,
                )
                agent_resp = await self.agent_service.run(agent_req)
                payload = agent_resp.model_dump()
            else:
                payload = {"error": "SQLDatabaseService and AgentService not configured."}

        elif intent == QueryIntent.AUTONOMOUS_AGENT:
            if self.agent_service:
                agent_req = AgentChatRequest(
                    query=query_text,
                    max_iterations=request.max_iterations or 6,
                )
                agent_resp = await self.agent_service.run(agent_req)
                payload = agent_resp.model_dump()
            else:
                payload = {"error": "AgentService not configured."}

        total_latency = round((time.perf_counter() - overall_start) * 1000, 2)
        return RouterDispatchResponse(
            query=query_text,
            intent=intent,
            confidence=confidence,
            strategy_used=strategy,
            reasoning=reasoning,
            latency_ms=total_latency,
            response=payload,
        )

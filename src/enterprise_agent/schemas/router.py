"""Request and response schemas for Semantic Query Router."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    """Categorized query execution intents."""

    DIRECT_CHAT = "direct_chat"
    RAG_SEARCH = "rag_search"
    SQL_DATABASE = "sql_database"
    AUTONOMOUS_AGENT = "autonomous_agent"


class RouterClassifyRequest(BaseModel):
    """Request payload to classify the intent of a query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user input query to analyze and route.",
        examples=["How many employees work in Engineering?"],
    )


class RouterClassifyResponse(BaseModel):
    """Routing classification decision with confidence and strategy metadata."""

    query: str = Field(..., description="Original user query.")
    intent: QueryIntent = Field(..., description="Classified execution target intent.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the routing classification (0.0 to 1.0).",
    )
    strategy_used: str = Field(
        ...,
        description="Routing method that produced the decision (heuristic, semantic, llm).",
    )
    reasoning: str = Field(
        ...,
        description="Brief justification explaining why this intent was selected.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Classification latency in milliseconds.",
    )


class RouterDispatchRequest(BaseModel):
    """Request payload to classify and immediately execute the appropriate subsystem."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user prompt to route and execute.",
        examples=["What is our remote work expense policy?"],
    )
    force_intent: QueryIntent | None = Field(
        default=None,
        description="Optional intent override bypassing the classifier.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional max reasoning steps if routed to Autonomous Agent.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional temperature override for LLM generation.",
    )


class RouterDispatchResponse(BaseModel):
    """End-to-end execution response with routing telemetry and subsystem payload."""

    query: str = Field(..., description="Original user query.")
    intent: QueryIntent = Field(..., description="Target intent that processed the request.")
    confidence: float = Field(..., description="Confidence score of the routing decision.")
    strategy_used: str = Field(..., description="Routing strategy that made the decision.")
    reasoning: str = Field(..., description="Justification for the chosen route.")
    latency_ms: float = Field(..., description="Total round-trip execution latency in ms.")
    response: Any = Field(..., description="Subsystem response payload (Chat/RAG/SQL/Agent).")

"""FastAPI Request and Response schemas for chat and health endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageInput(BaseModel):
    """Previous conversational turn in request history."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message role (user, assistant, or system)."
    )
    content: str = Field(..., min_length=1, description="Message text content.")


class ChatRequest(BaseModel):
    """User input payload for `/api/v1/chat`."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The prompt or business question from the user.",
        examples=["What is our standard vacation policy?"],
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional override for system instructions.",
    )
    history: list[ChatMessageInput] = Field(
        default_factory=list,
        description="Optional list of prior turns in the ongoing conversation.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional temperature override.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=8192,
        description="Optional max tokens limit.",
    )


class TokenUsageResponse(BaseModel):
    """Token consumption details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Standardized response payload from `/api/v1/chat`."""

    answer: str = Field(..., description="The assistant's generated response.")
    model: str = Field(..., description="The model that produced the answer.")
    usage: TokenUsageResponse = Field(..., description="Token breakdown.")
    latency_ms: float = Field(..., description="Round-trip generation latency in milliseconds.")
    status: Literal["success", "error"] = "success"


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    app_name: str
    version: str
    environment: str
    llm_healthy: bool

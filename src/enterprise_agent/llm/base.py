"""Abstract Base Class and data models for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Supported roles in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Standardized representation of a single conversation turn."""

    role: MessageRole
    content: str = Field(..., min_length=1, description="Text content of the message.")


class TokenUsage(BaseModel):
    """Token consumption accounting for the LLM request."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """Standardized output returned by any LLM provider."""

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str = "stop"


class LLMStreamChunk(BaseModel):
    """Single token chunk emitted during streaming responses."""

    delta_text: str
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract interface defining required behaviors for LLM implementations."""

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete text completion given a list of messages."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield streaming text chunks incrementally as they are generated."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity and authentication to upstream LLM service."""
        pass

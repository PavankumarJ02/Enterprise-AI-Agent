"""Mock LLM Provider for deterministic offline testing and local development."""

from collections.abc import AsyncIterator

from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)


class MockLLMProvider(LLMProvider):
    """Deterministic, mock LLM provider requiring no external network or API keys."""

    def __init__(
        self,
        default_response: str = "This is a simulated response from the enterprise MockLLMProvider.",
        model_name: str = "mock-enterprise-model",
    ) -> None:
        self.default_response = default_response
        self.model_name = model_name
        self.invocations: list[list[ChatMessage]] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Record invocation and return deterministic structured response."""
        self.invocations.append(messages)

        last_user_msg = ""
        for m in reversed(messages):
            if m.role.value == "user":
                last_user_msg = m.content
                break

        # Generate responsive reply for testing
        content = (
            f"[MOCK RESPONSE] Processed query: '{last_user_msg}'. Result: {self.default_response}"
        )

        usage = TokenUsage(
            prompt_tokens=len(last_user_msg.split()) * 2,
            completion_tokens=len(content.split()) * 2,
            total_tokens=(len(last_user_msg.split()) + len(content.split())) * 2,
        )

        return LLMResponse(
            content=content,
            model=self.model_name,
            usage=usage,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield simulated word-by-word streaming tokens."""
        self.invocations.append(messages)
        words = self.default_response.split(" ")
        for i, word in enumerate(words):
            yield LLMStreamChunk(
                delta_text=word + (" " if i < len(words) - 1 else ""),
                finish_reason=None,
            )
        yield LLMStreamChunk(delta_text="", finish_reason="stop")

    async def health_check(self) -> bool:
        """Always healthy in mock mode."""
        return True

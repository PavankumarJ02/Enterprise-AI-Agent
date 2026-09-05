"""OpenAI-compatible LLM Provider implementation."""

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from enterprise_agent.core.exceptions import (
    LLMAuthenticationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)

logger = get_logger(__name__)


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible client supporting OpenAI, Groq, Ollama, vLLM, and Azure."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert internal domain ChatMessage to OpenAI dict format."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def _map_exception(self, err: Exception) -> LLMException:
        """Map provider-specific exceptions to domain exceptions."""
        if isinstance(err, openai.AuthenticationError):
            return LLMAuthenticationError(f"LLM authentication failed: {err}")
        if isinstance(err, openai.RateLimitError):
            return LLMRateLimitError(f"LLM rate limit or quota exceeded: {err}")
        if isinstance(err, openai.APITimeoutError):
            return LLMTimeoutError(f"LLM request timed out: {err}")
        if isinstance(err, (openai.APIConnectionError, openai.APIStatusError, openai.APIError)):
            return LLMProviderError(f"LLM upstream service error: {err}")
        return LLMException(f"Unexpected error during LLM generation: {err}")

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response via OpenAI Chat Completions API."""
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens
        formatted_messages = self._format_messages(messages)

        try:
            raw_response = await self._client.chat.completions.create(
                model=self.default_model,
                messages=formatted_messages,  # type: ignore[arg-type]
                temperature=temp,
                max_tokens=tokens,
            )

            choice = raw_response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            usage = TokenUsage()
            if raw_response.usage:
                usage = TokenUsage(
                    prompt_tokens=raw_response.usage.prompt_tokens,
                    completion_tokens=raw_response.usage.completion_tokens,
                    total_tokens=raw_response.usage.total_tokens,
                )

            return LLMResponse(
                content=content,
                model=raw_response.model,
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error("Error invoking OpenAI-compatible provider: %s", e)
            raise self._map_exception(e) from e

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream response chunks via OpenAI streaming API."""
        temp = self.default_temperature if temperature is None else temperature
        tokens = self.default_max_tokens if max_tokens is None else max_tokens
        formatted_messages = self._format_messages(messages)

        try:
            stream_resp = await self._client.chat.completions.create(
                model=self.default_model,
                messages=formatted_messages,  # type: ignore[arg-type]
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )

            # Type guard for mypy
            if hasattr(stream_resp, "__aiter__"):
                async for chunk in stream_resp:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content or ""
                        finish = chunk.choices[0].finish_reason
                        if delta or finish:
                            yield LLMStreamChunk(delta_text=delta, finish_reason=finish)
        except Exception as e:
            logger.error("Error streaming from OpenAI-compatible provider: %s", e)
            raise self._map_exception(e) from e

    async def health_check(self) -> bool:
        """Verify endpoint connectivity with minimal ping query."""
        try:
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning("LLM health check failed: %s", e)
            return False

"""Unit tests for LLM abstraction layer."""

import openai
import pytest
from pydantic import SecretStr

try:
    import httpx2 as httpx
except ImportError:
    import httpx  # type: ignore[no-redef]

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from enterprise_agent.llm.base import ChatMessage, MessageRole
from enterprise_agent.llm.factory import get_llm_provider
from enterprise_agent.llm.mock_client import MockLLMProvider
from enterprise_agent.llm.openai_client import OpenAILLMProvider


@pytest.mark.asyncio
async def test_mock_llm_generate() -> None:
    """Verify mock LLM returns structured response with usage statistics."""
    provider = MockLLMProvider(
        default_response="Enterprise policy response.",
        model_name="mock-model",
    )
    messages = [ChatMessage(role=MessageRole.USER, content="What is the refund policy?")]

    response = await provider.generate(messages)

    assert response.model == "mock-model"
    assert "Enterprise policy response." in response.content
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    expected_total = response.usage.prompt_tokens + response.usage.completion_tokens
    assert response.usage.total_tokens == expected_total
    assert len(provider.invocations) == 1


@pytest.mark.asyncio
async def test_mock_llm_stream() -> None:
    """Verify mock LLM streaming yields chunks and final stop signal."""
    provider = MockLLMProvider(default_response="Alpha Beta Gamma")
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    chunks = []
    async for chunk in provider.stream(messages):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(c.delta_text for c in chunks)
    assert "Alpha Beta Gamma" in full_text
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_llm_health_check() -> None:
    """Verify mock health check passes."""
    provider = MockLLMProvider()
    assert await provider.health_check() is True


def test_factory_returns_mock() -> None:
    """Factory should instantiate MockLLMProvider when provider is 'mock'."""
    settings = Settings(llm_provider="mock")
    provider = get_llm_provider(settings)
    assert isinstance(provider, MockLLMProvider)


def test_factory_requires_api_key_for_openai() -> None:
    """Factory should raise ConfigurationError if OpenAI API key is missing."""
    settings = Settings(llm_provider="openai", llm_api_key=SecretStr(""))
    with pytest.raises(ConfigurationError) as exc_info:
        get_llm_provider(settings)
    assert "LLM_API_KEY" in str(exc_info.value)


def test_openai_exception_mapping() -> None:
    """Ensure provider maps OpenAI SDK exceptions to domain exceptions."""
    provider = OpenAILLMProvider(api_key="sk-test", default_model="gpt-4o-mini")

    dummy_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    dummy_response = httpx.Response(401, request=dummy_request)

    # 1. Authentication Error
    auth_err = openai.AuthenticationError(
        message="Invalid API Key",
        response=dummy_response,
        body={"error": {"message": "Invalid API Key"}},
    )
    mapped_auth = provider._map_exception(auth_err)
    assert isinstance(mapped_auth, LLMAuthenticationError)

    # 2. Rate Limit Error
    rl_response = httpx.Response(429, request=dummy_request)
    rl_err = openai.RateLimitError(
        message="Rate limit hit",
        response=rl_response,
        body={"error": {"message": "Rate limit"}},
    )
    mapped_rl = provider._map_exception(rl_err)
    assert isinstance(mapped_rl, LLMRateLimitError)

    # 3. Timeout Error
    timeout_err = openai.APITimeoutError(request=dummy_request)
    mapped_timeout = provider._map_exception(timeout_err)
    assert isinstance(mapped_timeout, LLMTimeoutError)

    # 4. Connection / Server Error
    conn_err = openai.APIConnectionError(request=dummy_request)
    mapped_conn = provider._map_exception(conn_err)
    assert isinstance(mapped_conn, LLMProviderError)

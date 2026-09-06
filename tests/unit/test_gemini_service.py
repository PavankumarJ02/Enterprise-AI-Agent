"""Unit tests for Gemini LLM Provider and conversion logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors
from pydantic import SecretStr

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
from enterprise_agent.llm.gemini_client import GeminiLLMProvider


def test_gemini_convert_messages() -> None:
    """Verify system instruction is extracted and assistant role is mapped to 'model'."""
    provider = GeminiLLMProvider(api_key="fake-key", default_model="gemini-2.5-flash")

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a strict compliance officer."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Greetings, employee."),
        ChatMessage(role=MessageRole.USER, content="What is the travel expense limit?"),
    ]

    system_instruction, contents = provider._convert_messages(messages)

    assert system_instruction == "You are a strict compliance officer."
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts is not None
    assert contents[0].parts[0].text == "Hello"
    assert contents[1].role == "model"  # Gemini requirement!
    assert contents[1].parts is not None
    assert contents[1].parts[0].text == "Greetings, employee."
    assert contents[2].role == "user"
    assert contents[2].parts is not None
    assert contents[2].parts[0].text == "What is the travel expense limit?"


def test_gemini_convert_messages_fallback_when_empty() -> None:
    """Ensure empty input provides a valid fallback user content part."""
    provider = GeminiLLMProvider(api_key="fake-key")
    system_instruction, contents = provider._convert_messages([])
    assert system_instruction is None
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_gemini_exception_mapping() -> None:
    """Verify google-genai APIError codes map to application domain exceptions."""
    provider = GeminiLLMProvider(api_key="fake-key")

    # 1. 401 / 403 Authentication Error
    auth_err = errors.APIError(401, "API_KEY_INVALID")
    mapped_auth = provider._map_exception(auth_err)
    assert isinstance(mapped_auth, LLMAuthenticationError)

    # 2. 429 Rate Limit
    rl_err = errors.APIError(429, "Resource has been exhausted (quota)")
    mapped_rl = provider._map_exception(rl_err)
    assert isinstance(mapped_rl, LLMRateLimitError)

    # 3. 504 Timeout
    timeout_err = errors.APIError(504, "Deadline Exceeded")
    mapped_timeout = provider._map_exception(timeout_err)
    assert isinstance(mapped_timeout, LLMTimeoutError)

    # 4. 500 Server Error
    server_err = errors.APIError(500, "Internal error")
    mapped_server = provider._map_exception(server_err)
    assert isinstance(mapped_server, LLMProviderError)


def test_factory_instantiates_gemini_provider() -> None:
    """Ensure factory instantiates GeminiLLMProvider when configured."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key=SecretStr("valid-gemini-key"),
        llm_model="gemini-2.5-flash",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.default_model == "gemini-2.5-flash"


def test_factory_gemini_fallback_to_llm_api_key() -> None:
    """Ensure factory falls back to llm_api_key if gemini_api_key is omitted."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key=SecretStr(""),
        llm_api_key=SecretStr("fallback-key"),
        llm_model="gemini-2.5-flash",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GeminiLLMProvider)


def test_factory_gemini_missing_key_raises() -> None:
    """Ensure factory raises ConfigurationError if no Gemini key is provided."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key=SecretStr(""),
        llm_api_key=SecretStr(""),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        get_llm_provider(settings)
    assert "GEMINI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_generate_success() -> None:
    """Mock test for successful Gemini generation output formatting."""
    provider = GeminiLLMProvider(api_key="fake-key", max_retries=1)

    # Mock response structure returned by client.aio.models.generate_content
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 15
    mock_usage.candidates_token_count = 35
    mock_usage.total_token_count = 50

    mock_raw_response = MagicMock()
    mock_raw_response.text = "Gemini synthesized enterprise response."
    mock_raw_response.candidates = [mock_candidate]
    mock_raw_response.usage_metadata = mock_usage

    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(return_value=mock_raw_response)

    messages = [ChatMessage(role=MessageRole.USER, content="Hello Gemini")]
    result = await provider.generate(messages)

    assert result.content == "Gemini synthesized enterprise response."
    assert result.model == "gemini-2.5-flash"
    assert result.usage.total_tokens == 50
    assert result.finish_reason == "stop"

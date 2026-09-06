"""Unit tests for LLMRouter zero-shot intent classifier."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.llm.base import LLMProvider, LLMResponse, TokenUsage
from enterprise_agent.router.llm import LLMRouter
from enterprise_agent.schemas.router import QueryIntent


def _mock_resp(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="mock-router-model",
        usage=TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_llm_router_parses_clean_json() -> None:
    """Verify clean JSON string parses correctly into QueryIntent and confidence."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_resp(
        '{"intent": "sql_database", "confidence": 0.92, '
        '"reasoning": "Requires querying employee tables."}'
    )

    router = LLMRouter(llm=mock_llm)
    intent, conf, reasoning = await router.classify("What is the average engineering salary?")

    assert intent == QueryIntent.SQL_DATABASE
    assert conf == 0.92
    assert "employee tables" in reasoning


@pytest.mark.asyncio
async def test_llm_router_strips_markdown_code_fences() -> None:
    """Verify markdown code blocks wrapping JSON are cleanly stripped."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_resp(
        "```json\n"
        '{\n  "intent": "autonomous_agent",\n  "confidence": 0.96,\n'
        '  "reasoning": "Compound calculation task."\n}\n'
        "```"
    )

    router = LLMRouter(llm=mock_llm)
    intent, conf, _ = await router.classify("Find policy and calculate per diem")

    assert intent == QueryIntent.AUTONOMOUS_AGENT
    assert conf == 0.96


@pytest.mark.asyncio
async def test_llm_router_regex_fallback_on_unformatted_text() -> None:
    """Verify unformatted prose containing intent token falls back to regex extraction."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_resp(
        "Based on my evaluation, this request represents a rag_search task for internal docs."
    )

    router = LLMRouter(llm=mock_llm)
    intent, conf, _ = await router.classify("Tell me about company holiday schedule")

    assert intent == QueryIntent.RAG_SEARCH
    assert conf >= 0.70


@pytest.mark.asyncio
async def test_llm_router_defaults_on_total_garbage() -> None:
    """Verify unparseable response defaults safely to DIRECT_CHAT."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_resp("Unrecognized output with no intent.")

    router = LLMRouter(llm=mock_llm)
    intent, conf, reasoning = await router.classify("Random text")

    assert intent == QueryIntent.DIRECT_CHAT
    assert conf == 0.50
    assert "Default fallback" in reasoning

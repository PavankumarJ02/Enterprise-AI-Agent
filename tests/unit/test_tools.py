"""Unit tests for Agent Tools (Calculator, Clock, KnowledgeSearch, Registry)."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import RerankResponse, RerankResultItem
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.clock import CurrentTimeTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.tools.search import KnowledgeSearchTool


@pytest.mark.asyncio
async def test_calculator_basic_arithmetic() -> None:
    """Verify CalculatorTool correctly computes standard math expressions."""
    calc = CalculatorTool()

    r1 = await calc.execute(expression="75 * 5")
    assert not r1.is_error
    assert r1.output == "375"

    r2 = await calc.execute(expression="(1200 + 300) / 1.5")
    assert not r2.is_error
    assert r2.output == "1000"

    r3 = await calc.execute(expression="2 ** 10")
    assert not r3.is_error
    assert r3.output == "1024"


@pytest.mark.asyncio
async def test_calculator_safe_functions() -> None:
    """Verify CalculatorTool correctly evaluates safe whitelisted functions."""
    calc = CalculatorTool()

    r1 = await calc.execute(expression="round(15.6789, 2)")
    assert not r1.is_error
    assert r1.output == "15.68"

    r2 = await calc.execute(expression="min(45, 12, 89)")
    assert not r2.is_error
    assert r2.output == "12"

    r3 = await calc.execute(expression="sum([10, 20, 30])")
    assert not r3.is_error
    assert r3.output == "60"


@pytest.mark.asyncio
async def test_calculator_errors_and_security() -> None:
    """Verify CalculatorTool rejects division by zero and unauthorized code execution."""
    calc = CalculatorTool()

    # Division by zero
    r_div0 = await calc.execute(expression="100 / 0")
    assert r_div0.is_error
    assert "Division by zero" in r_div0.output

    # Empty expression
    r_empty = await calc.execute(expression="")
    assert r_empty.is_error

    # Security injection attempts
    r_sec1 = await calc.execute(expression="__import__('os').system('dir')")
    assert r_sec1.is_error

    r_sec2 = await calc.execute(expression="open('README.md').read()")
    assert r_sec2.is_error


@pytest.mark.asyncio
async def test_current_time_tool() -> None:
    """Verify CurrentTimeTool returns valid ISO and weekday metadata."""
    clock = CurrentTimeTool()
    res = await clock.execute()

    assert not res.is_error
    assert "Current Date & Time" in res.output
    assert "iso" in res.metadata
    assert "weekday" in res.metadata
    assert res.metadata["year"] >= 2026


def test_tool_registry_management() -> None:
    """Verify ToolRegistry registers, retrieves, and formats tool specifications."""
    registry = ToolRegistry()
    calc = CalculatorTool()
    clock = CurrentTimeTool()

    registry.register(calc)
    registry.register(clock)

    # Retrieval
    assert registry.get("calculator") is calc
    assert registry.get("CALCULATOR") is calc  # case-insensitive
    assert registry.get("current_time") is clock
    assert registry.get("unknown_tool") is None

    # Schemas
    schemas = registry.get_schemas()
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"calculator", "current_time"}

    # Prompt formatting
    prompt_text = registry.format_prompt_description()
    assert "- Tool: calculator" in prompt_text
    assert "- Tool: current_time" in prompt_text


@pytest.mark.asyncio
async def test_tool_registry_safe_execution() -> None:
    """Verify ToolRegistry dispatches execution and handles unknown tool gracefully."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    res_valid = await registry.execute("calculator", {"expression": "50 + 25"})
    assert not res_valid.is_error
    assert res_valid.output == "75"

    res_unknown = await registry.execute("imaginary_tool", {})
    assert res_unknown.is_error
    assert "not found" in res_unknown.output


@pytest.mark.asyncio
async def test_knowledge_search_tool() -> None:
    """Verify KnowledgeSearchTool delegates to two-stage retrieval and formats output."""
    mock_two_stage = AsyncMock(spec=TwoStageRetrievalService)
    item = RerankResultItem(
        chunk_id="c1",
        document_id="d1",
        content="Employee business travel meal allowance is $75 per day.",
        score=0.8,
        rerank_score=0.985,
        initial_rank=1,
        final_rank=1,
        initial_score=0.8,
        chunk_index=0,
        metadata={"source": "travel_policy.md"},
    )
    mock_two_stage.retrieve_and_rerank.return_value = RerankResponse(
        query="meal allowance",
        total_candidates=1,
        total_results=1,
        retrieval_strategy="hybrid",
        model="test-reranker",
        results=[item],
    )

    tool = KnowledgeSearchTool(two_stage_service=mock_two_stage)
    res = await tool.execute(query="meal allowance", top_k=2)

    assert not res.is_error
    assert "Source: travel_policy.md" in res.output
    assert "$75 per day" in res.output
    mock_two_stage.retrieve_and_rerank.assert_awaited_once()

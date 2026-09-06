"""Unit tests for the Agent ReAct parsing and AgentService execution loop."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.agent.parser import parse_agent_output
from enterprise_agent.agent.service import AgentService
from enterprise_agent.llm.base import LLMProvider, LLMResponse, TokenUsage
from enterprise_agent.schemas.agent import AgentChatRequest
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.registry import ToolRegistry


def _mock_llm_resp(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="mock-agent-llm",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
        finish_reason="stop",
    )


def test_parse_agent_output_action() -> None:
    """Verify parse_agent_output correctly extracts thought, action, and JSON input."""
    raw = (
        "Thought: I need to calculate the total allowance.\n"
        "Action: calculator\n"
        'Action Input: {"expression": "75 * 6"}\n'
    )
    parsed = parse_agent_output(raw)
    assert not parsed.is_final
    assert "calculate the total allowance" in parsed.thought
    assert parsed.action == "calculator"
    assert parsed.action_input == {"expression": "75 * 6"}


def test_parse_agent_output_action_markdown_json() -> None:
    """Verify parse_agent_output strips markdown code fence around JSON action input."""
    raw = (
        "Thought: Evaluating calculation.\n"
        "Action: calculator\n"
        "Action Input: ```json\n"
        '{\n  "expression": "100 + 250"\n}\n'
        "```\n"
    )
    parsed = parse_agent_output(raw)
    assert not parsed.is_final
    assert parsed.action == "calculator"
    assert parsed.action_input == {"expression": "100 + 250"}


def test_parse_agent_output_final_answer() -> None:
    """Verify parse_agent_output extracts final answer and preceding thought."""
    raw = (
        "Thought: I have all required numbers.\n"
        "Final Answer: The total travel reimbursement for 6 days is $450."
    )
    parsed = parse_agent_output(raw)
    assert parsed.is_final
    assert parsed.final_answer == "The total travel reimbursement for 6 days is $450."
    assert "I have all required numbers" in parsed.thought


def test_parse_agent_output_fallback_direct_text() -> None:
    """Verify parse_agent_output treats plain text without tags as direct final answer."""
    raw = "The company headquarter is located in San Francisco, California."
    parsed = parse_agent_output(raw)
    assert parsed.is_final
    assert parsed.final_answer == raw


@pytest.mark.asyncio
async def test_agent_service_direct_final_answer() -> None:
    """Verify AgentService handles queries requiring zero tool calls immediately."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_resp(
        "Thought: No tools needed.\nFinal Answer: Hello! How can I assist you today?"
    )

    registry = ToolRegistry()
    service = AgentService(llm=mock_llm, tool_registry=registry)

    req = AgentChatRequest(query="Hello")
    res = await service.run(req)

    assert res.status == "success"
    assert res.final_answer == "Hello! How can I assist you today?"
    assert res.iterations == 0
    assert len(res.steps) == 0


@pytest.mark.asyncio
async def test_agent_service_single_tool_execution() -> None:
    """Verify AgentService executes single tool call and synthesizes final answer."""
    mock_llm = AsyncMock(spec=LLMProvider)
    # Turn 1: Action -> calculator
    # Turn 2: Final Answer
    mock_llm.generate.side_effect = [
        _mock_llm_resp(
            "Thought: I will calculate 75 * 5.\n"
            "Action: calculator\n"
            'Action Input: {"expression": "75 * 5"}'
        ),
        _mock_llm_resp(
            "Thought: I have the calculation result.\n"
            "Final Answer: The total meal allowance is $375."
        ),
    ]

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    service = AgentService(llm=mock_llm, tool_registry=registry)

    req = AgentChatRequest(query="What is 75 * 5?")
    res = await service.run(req)

    assert res.status == "success"
    assert res.final_answer == "The total meal allowance is $375."
    assert res.iterations == 1
    assert len(res.steps) == 1
    assert res.steps[0].action == "calculator"
    assert res.steps[0].observation == "375"


@pytest.mark.asyncio
async def test_agent_service_max_iterations_termination() -> None:
    """Verify AgentService terminates safely when max_iterations limit is reached."""
    mock_llm = AsyncMock(spec=LLMProvider)
    # Always emit tool actions without terminating
    mock_llm.generate.return_value = _mock_llm_resp(
        'Thought: Keep looping.\nAction: calculator\nAction Input: {"expression": "1 + 1"}'
    )

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    service = AgentService(llm=mock_llm, tool_registry=registry, default_max_iterations=3)

    req = AgentChatRequest(query="Infinite loop task", max_iterations=2)
    res = await service.run(req)

    assert res.status == "max_iterations_reached"
    assert res.iterations == 2
    assert len(res.steps) == 2

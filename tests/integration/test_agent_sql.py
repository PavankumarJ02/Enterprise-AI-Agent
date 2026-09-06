"""Integration test demonstrating ReAct Agent using SQL tooling and schema inspection."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from enterprise_agent.agent.service import AgentService
from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)
from enterprise_agent.schemas.agent import AgentChatRequest
from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.service import SQLDatabaseService
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.tools.sql import SQLQueryTool, SQLSchemaTool


class MultiTurnMockLLM(LLMProvider):
    """Mock LLM sequentially returning predefined ReAct reasoning steps."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        idx = min(self.call_count, len(self.responses) - 1)
        resp_text = self.responses[idx]
        self.call_count += 1
        return LLMResponse(
            content=resp_text,
            model="agent-sql-mock",
            usage=TokenUsage(prompt_tokens=25, completion_tokens=20, total_tokens=45),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(delta_text="")

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_react_sql_multistep_workflow(tmp_path: Path) -> None:
    """Verify agent inspects schema, executes SQL aggregation, and delivers final answer."""
    db_file = tmp_path / "agent_sql.db"
    manager = SQLiteManager(db_path=db_file)
    sql_service = SQLDatabaseService(manager=manager)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SQLSchemaTool(sql_service=sql_service))
    registry.register(SQLQueryTool(sql_service=sql_service))

    mock_llm = MultiTurnMockLLM(
        responses=[
            # Step 1: Introspect schema
            (
                "Thought: I need to verify the column names of the products table.\n"
                "Action: sql_schema\n"
                'Action Input: {"tables": ["products"]}'
            ),
            # Step 2: Query inventory sum
            (
                "Thought: Now I will query the sum of price * stock_quantity for Hardware.\n"
                "Action: sql_query\n"
                'Action Input: {"query": "SELECT SUM(price * stock_quantity) AS total_val '
                "FROM products WHERE category = 'Hardware'\"}"
            ),
            # Step 3: Deliver final answer
            (
                "Thought: I have the hardware total inventory value from the database query.\n"
                "Final Answer: The total inventory value for hardware products is $1,408,500.00."
            ),
        ]
    )

    service = AgentService(llm=mock_llm, tool_registry=registry)
    request = AgentChatRequest(
        query="What is the total inventory value for hardware products?",
        max_iterations=5,
    )

    response = await service.run(request)

    assert response.status == "success"
    assert response.iterations == 2  # 2 tool actions executed
    assert len(response.steps) == 2

    # Step 1 verification: sql_schema
    assert response.steps[0].action == "sql_schema"
    assert "stock_quantity" in response.steps[0].observation

    # Step 2 verification: sql_query
    assert response.steps[1].action == "sql_query"
    assert "1408500" in response.steps[1].observation

    # Final answer verification
    assert "$1,408,500.00" in response.final_answer

"""Thread-safe Tool Registry for dynamic registration, schema export, and execution."""

import asyncio
import json
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry maintaining available agent tools and handling execution dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry."""
        normalized_name = tool.name.strip().lower()
        if normalized_name in self._tools:
            logger.warning("Overwriting existing tool '%s' in ToolRegistry", normalized_name)
        self._tools[normalized_name] = tool
        logger.info("Registered tool '%s' in ToolRegistry", normalized_name)

    def get(self, name: str) -> BaseTool | None:
        """Retrieve tool by identifier name."""
        return self._tools.get(name.strip().lower())

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Return all registered tool identifier names."""
        return list(self._tools.keys())

    def get_schemas(self, allowed_tools: list[str] | None = None) -> list[dict[str, Any]]:
        """Export tool specifications as list of JSON schemas."""
        allowed = {t.lower() for t in allowed_tools} if allowed_tools else None
        schemas: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            if allowed is None or name in allowed:
                schemas.append(tool.to_schema_dict())
        return schemas

    def format_prompt_description(self, allowed_tools: list[str] | None = None) -> str:
        """Format tools into structured text suitable for ReAct system prompts."""
        schemas = self.get_schemas(allowed_tools=allowed_tools)
        if not schemas:
            return "No tools available."

        tool_blocks: list[str] = []
        for s in schemas:
            params_json = json.dumps(s["parameters"], indent=2)
            block = (
                f"- Tool: {s['name']}\n"
                f"  Description: {s['description']}\n"
                f"  Parameters Schema (JSON):\n{params_json}"
            )
            tool_blocks.append(block)

        return "\n\n".join(tool_blocks)

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Safely look up and execute a tool with the provided arguments."""
        normalized_name = name.strip().lower()
        tool = self.get(normalized_name)
        if not tool:
            available = list(self._tools.keys())
            return ToolResult(
                output=f"Error: Tool '{name}' not found. Available tools: {available}",
                is_error=True,
            )

        try:
            return await tool.execute(**args)
        except Exception as exc:
            logger.error("Unhandled exception executing tool '%s': %s", name, exc)
            return ToolResult(
                output=f"Error executing tool '{name}': {exc}",
                is_error=True,
            )

    async def execute_many(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolResult]:
        """Execute multiple tool invocations concurrently via asyncio.gather."""
        if not tool_calls:
            return []

        tasks = [self.execute(name, args) for name, args in tool_calls]
        return list(await asyncio.gather(*tasks))

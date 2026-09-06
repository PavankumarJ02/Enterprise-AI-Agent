"""Abstract Base Class and data models for Agent Tools."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Encapsulates the output and execution telemetry of an agent tool."""

    output: str = Field(
        ...,
        description="String output or representation returned by the tool.",
    )
    is_error: bool = Field(
        default=False,
        description="Whether the tool encountered an execution error.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured telemetry or raw metadata returned by the tool.",
    )


class BaseTool(ABC):
    """Abstract contract for tools invokable by the autonomous agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the tool (alphanumeric and underscores only)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Clear description of the tool's purpose, capabilities, and expected usage."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining the accepted arguments, types, and required fields."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with supplied keyword arguments and return a ToolResult.

        Args:
            **kwargs: Arguments conforming to parameters_schema.

        Returns:
            ToolResult containing stringified output and error telemetry.
        """
        ...

    def to_schema_dict(self) -> dict[str, Any]:
        """Export tool contract in standard format for LLM prompting."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

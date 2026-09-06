"""Pydantic schemas for the Tool-Augmented Agent Engine."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from enterprise_agent.schemas.chat import ChatMessageInput


class ToolDefinitionResponse(BaseModel):
    """Schema describing an available tool registered in the agent's tool registry."""

    name: str = Field(..., description="Unique tool name.")
    description: str = Field(..., description="Description of the tool's purpose and usage.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema specifying tool parameter names, types, and constraints.",
    )


class AgentStep(BaseModel):
    """Telemetry representation of a single ReAct reasoning and execution step."""

    step_number: int = Field(..., ge=1, description="Sequential 1-based step index.")
    thought: str = Field(..., description="Agent reasoning explaining the rationale for this step.")
    action: str = Field(..., description="Name of the selected tool to execute.")
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool.",
    )
    observation: str = Field(..., description="Output or error returned by the executed tool.")
    duration_ms: float = Field(default=0.0, description="Execution duration of the tool in ms.")


class AgentChatRequest(BaseModel):
    """Payload for invoking the autonomous tool-augmented agent."""

    query: str = Field(
        ...,
        min_length=1,
        description="User natural language task or question.",
        examples=[
            "What is our daily travel meal allowance, and what would the total be for 5 days?"
        ],
    )
    conversation_history: list[ChatMessageInput] = Field(
        default_factory=list,
        description="Optional preceding conversation turns for context.",
    )
    max_iterations: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Safety ceiling on ReAct reasoning steps before halting.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional whitelist restricting which registered tools may be called.",
    )


class AgentChatResponse(BaseModel):
    """Response payload containing final synthesis and full intermediate reasoning steps."""

    query: str = Field(..., description="Original user query.")
    final_answer: str = Field(..., description="Synthesized final answer to the user.")
    steps: list[AgentStep] = Field(
        default_factory=list,
        description="Sequential list of thoughts, tool actions, and observations executed.",
    )
    iterations: int = Field(..., description="Total ReAct steps executed.")
    total_duration_ms: float = Field(..., description="Total agent execution duration in ms.")
    status: Literal["success", "max_iterations_reached", "error"] = Field(
        default="success",
        description="Final execution outcome status.",
    )

"""API routes for the Tool-Augmented Autonomous Agent."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from enterprise_agent.agent.service import AgentService
from enterprise_agent.api.deps import get_agent_service, get_tool_registry
from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    ToolDefinitionResponse,
)
from enterprise_agent.tools.registry import ToolRegistry

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["Autonomous Agent"])


@router.get(
    "/tools",
    response_model=list[ToolDefinitionResponse],
    status_code=status.HTTP_200_OK,
    summary="List available agent tools",
    description=(
        "Returns list of registered tools with their names, descriptions, and JSON Schemas."
    ),
)
async def list_tools(
    tool_registry: ToolRegistry = Depends(get_tool_registry),
) -> list[ToolDefinitionResponse]:
    """Retrieve specifications for all registered agent tools."""
    schemas = tool_registry.get_schemas()
    return [
        ToolDefinitionResponse(
            name=s["name"],
            description=s["description"],
            parameters=s["parameters"],
        )
        for s in schemas
    ]


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-step ReAct agent task",
    description=(
        "Executes iterative ReAct reasoning and tool invocation loop, "
        "returning final answer with intermediate thoughts, actions, and observations."
    ),
)
async def agent_chat(
    request: AgentChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentChatResponse:
    """Run autonomous multi-turn agent task synchronously."""
    try:
        response = await service.run(request)
        return response
    except Exception as exc:
        logger.error("Error during agent execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        ) from exc


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream ReAct agent execution steps (SSE)",
    description=(
        "Streams real-time Server-Sent Events for each reasoning step: "
        "thoughts, tool invocations, tool outputs, and final answer."
    ),
)
async def agent_stream(
    request: AgentChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    """Stream autonomous agent reasoning steps via Server-Sent Events."""
    try:
        generator = service.run_stream(request)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        logger.error("Error setting up agent streaming: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent streaming setup failed: {exc}",
        ) from exc

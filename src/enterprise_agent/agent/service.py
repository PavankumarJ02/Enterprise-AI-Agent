"""Agent execution service managing the ReAct loop, tool dispatch, and streaming."""

import json
import time
from collections.abc import AsyncIterator

from enterprise_agent.agent.parser import parse_agent_output
from enterprise_agent.agent.prompts import REACT_SYSTEM_PROMPT
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.schemas.agent import AgentChatRequest, AgentChatResponse, AgentStep
from enterprise_agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class AgentService:
    """Orchestrates autonomous multi-turn ReAct reasoning and tool invocation."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        default_max_iterations: int = 6,
        default_timeout_seconds: float = 60.0,
        default_temperature: float = 0.1,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.default_max_iterations = default_max_iterations
        self.default_timeout_seconds = default_timeout_seconds
        self.default_temperature = default_temperature

    def _build_initial_messages(self, request: AgentChatRequest) -> list[ChatMessage]:
        """Construct system prompt with tool schemas and conversation history."""
        tools_description = self.tool_registry.format_prompt_description(
            allowed_tools=request.allowed_tools
        )
        system_content = REACT_SYSTEM_PROMPT.format(tools_description=tools_description)

        messages: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content=system_content)]

        # Append prior conversation context if provided
        for msg in request.conversation_history:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.content))

        # Append user query
        messages.append(ChatMessage(role=MessageRole.USER, content=request.query))
        return messages

    async def run(self, request: AgentChatRequest) -> AgentChatResponse:
        """Execute autonomous ReAct loop synchronously, returning final answer and telemetry."""
        start_time = time.perf_counter()
        max_steps = request.max_iterations or self.default_max_iterations
        messages = self._build_initial_messages(request)
        steps: list[AgentStep] = []

        logger.info(
            "Starting Agent execution for query: '%s' (max_steps=%d)",
            request.query,
            max_steps,
        )

        for step_idx in range(1, max_steps + 1):
            elapsed_sec = time.perf_counter() - start_time
            if elapsed_sec > self.default_timeout_seconds:
                logger.warning(
                    "Agent execution exceeded timeout of %.1fs", self.default_timeout_seconds
                )
                break

            # 1. Ask LLM for next reasoning step
            llm_resp = await self.llm.generate(
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=1024,
            )

            # 2. Parse Thought, Action, or Final Answer
            parsed = parse_agent_output(llm_resp.content)

            # If agent signaled completion
            if parsed.is_final:
                total_duration = round((time.perf_counter() - start_time) * 1000, 2)
                logger.info(
                    "Agent finished in %d steps (duration=%.2fms)",
                    len(steps),
                    total_duration,
                )
                return AgentChatResponse(
                    query=request.query,
                    final_answer=parsed.final_answer,
                    steps=steps,
                    iterations=len(steps),
                    total_duration_ms=total_duration,
                    status="success",
                )

            # 3. Execute Tool Action
            tool_start = time.perf_counter()
            tool_result = await self.tool_registry.execute(
                name=parsed.action,
                args=parsed.action_input or {},
            )
            tool_duration = round((time.perf_counter() - tool_start) * 1000, 2)

            step = AgentStep(
                step_number=step_idx,
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input or {},
                observation=tool_result.output,
                duration_ms=tool_duration,
            )
            steps.append(step)

            # 4. Append Scratchpad turns for next LLM iteration
            assistant_turn = (
                f"Thought: {parsed.thought}\n"
                f"Action: {parsed.action}\n"
                f"Action Input: {json.dumps(parsed.action_input or {})}"
            )
            observation_turn = f"Observation: {tool_result.output}"

            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_turn))
            messages.append(ChatMessage(role=MessageRole.USER, content=observation_turn))

        # Max iterations reached without explicit final answer
        logger.warning(
            "Agent reached maximum iterations (%d) without explicit final answer", max_steps
        )
        total_duration = round((time.perf_counter() - start_time) * 1000, 2)

        # Prompt for a forced final summary given accumulated observations
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "You have reached the maximum allowed steps. "
                    "Based on all observations above, provide your Final Answer now."
                ),
            )
        )
        fallback_resp = await self.llm.generate(
            messages=messages,
            temperature=self.default_temperature,
            max_tokens=512,
        )
        parsed_fallback = parse_agent_output(fallback_resp.content)
        final_answer = parsed_fallback.final_answer or fallback_resp.content.strip()

        return AgentChatResponse(
            query=request.query,
            final_answer=final_answer,
            steps=steps,
            iterations=len(steps),
            total_duration_ms=total_duration,
            status="max_iterations_reached",
        )

    async def run_stream(self, request: AgentChatRequest) -> AsyncIterator[str]:
        """Stream ReAct agent steps and final answer via Server-Sent Events (SSE)."""
        start_time = time.perf_counter()
        max_steps = request.max_iterations or self.default_max_iterations
        messages = self._build_initial_messages(request)

        for step_idx in range(1, max_steps + 1):
            if time.perf_counter() - start_time > self.default_timeout_seconds:
                yield f"event: error\ndata: {json.dumps({'error': 'AGENT_TIMEOUT'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            llm_resp = await self.llm.generate(
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=1024,
            )
            parsed = parse_agent_output(llm_resp.content)

            # Emit thought event
            if parsed.thought:
                t_payload = json.dumps({"step": step_idx, "thought": parsed.thought})
                yield f"event: thought\ndata: {t_payload}\n\n"

            if parsed.is_final:
                ans_payload = json.dumps({"answer": parsed.final_answer})
                yield f"event: final_answer\ndata: {ans_payload}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Emit tool call event
            tc_payload = json.dumps(
                {"step": step_idx, "tool": parsed.action, "input": parsed.action_input or {}}
            )
            yield f"event: tool_call\ndata: {tc_payload}\n\n"

            # Execute tool
            tool_res = await self.tool_registry.execute(parsed.action, parsed.action_input or {})

            # Emit tool result event
            tr_payload = json.dumps(
                {
                    "step": step_idx,
                    "tool": parsed.action,
                    "output": tool_res.output,
                    "is_error": tool_res.is_error,
                }
            )
            yield f"event: tool_result\ndata: {tr_payload}\n\n"

            # Append scratchpad
            assistant_turn = (
                f"Thought: {parsed.thought}\n"
                f"Action: {parsed.action}\n"
                f"Action Input: {json.dumps(parsed.action_input or {})}"
            )
            observation_turn = f"Observation: {tool_res.output}"
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_turn))
            messages.append(ChatMessage(role=MessageRole.USER, content=observation_turn))

        max_payload = json.dumps({"answer": "Maximum reasoning steps reached."})
        yield f"event: final_answer\ndata: {max_payload}\n\n"
        yield "data: [DONE]\n\n"

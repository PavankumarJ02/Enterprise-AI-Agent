"""Deterministic parsing of ReAct agent reasoning steps and tool calls."""

import json
import re
from dataclasses import dataclass
from typing import Any

from enterprise_agent.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedAgentResponse:
    """Represents the structured interpretation of an LLM generation step."""

    thought: str
    is_final: bool
    final_answer: str = ""
    action: str = ""
    action_input: dict[str, Any] | None = None


def parse_agent_output(text: str) -> ParsedAgentResponse:
    """Parse raw LLM output into thought, tool action/input, or final answer.

    Handles standard ReAct formatting, JSON action blocks, and graceful fallbacks.
    """
    clean_text = text.strip()

    # 1. Check for Final Answer
    final_answer_match = re.search(r"Final Answer:\s*(.+)", clean_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        # Extract preceding thought if present
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=Final Answer:)", clean_text, re.IGNORECASE | re.DOTALL
        )
        thought = (
            thought_match.group(1).strip()
            if thought_match
            else "I have gathered enough information to answer."
        )
        return ParsedAgentResponse(
            thought=thought,
            is_final=True,
            final_answer=final_answer,
        )

    # 2. Check for Action and Action Input
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", clean_text, re.IGNORECASE)
    input_match = re.search(
        r"Action Input:\s*(.+?)(?=(?:Observation:|Thought:|$))",
        clean_text,
        re.IGNORECASE | re.DOTALL,
    )

    if action_match:
        action = action_match.group(1).strip()

        # Extract thought preceding action
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=Action:)", clean_text, re.IGNORECASE | re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else f"Invoking tool {action}"

        raw_input = input_match.group(1).strip() if input_match else "{}"

        # Clean markdown codeblocks if LLM enclosed JSON in ```json ... ```
        if raw_input.startswith("```"):
            raw_input = re.sub(r"^```(?:json)?\s*", "", raw_input)
            raw_input = re.sub(r"\s*```$", "", raw_input)

        parsed_input: dict[str, Any]
        try:
            val = json.loads(raw_input)
            if isinstance(val, dict):
                parsed_input = val
            else:
                parsed_input = {"input": val}
        except Exception:
            # If not valid JSON, treat raw string as an 'input' or 'query' or 'expression' param
            parsed_input = {"input": raw_input}

        return ParsedAgentResponse(
            thought=thought,
            is_final=False,
            action=action,
            action_input=parsed_input,
        )

    # 3. Fallback: If no Action and no explicit Final Answer tag, treat entire text as final answer
    logger.info(
        "No Action or Final Answer tag detected in agent output; treating text as final answer."
    )
    thought_match = re.search(r"Thought:\s*(.*?)(?=\n\n|$)", clean_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "Formulating direct answer."
    # Strip leading "Thought: ..." if present from final answer
    final_ans = re.sub(r"^Thought:.*?\n\n", "", clean_text, flags=re.DOTALL).strip()
    if not final_ans:
        final_ans = clean_text

    return ParsedAgentResponse(
        thought=thought,
        is_final=True,
        final_answer=final_ans,
    )

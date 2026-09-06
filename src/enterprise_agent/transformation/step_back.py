"""Step-Back Prompting query transformer implementation."""

import re
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.transformation.base import QueryTransformer
from enterprise_agent.transformation.prompts import (
    STEP_BACK_SYSTEM_PROMPT,
    STEP_BACK_USER_PROMPT,
)

logger = get_logger(__name__)


class StepBackTransformer(QueryTransformer):
    """Abstracts specific detailed queries into broader foundational or architectural questions."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def generate_step_back(self, query: str) -> tuple[str, str | None]:
        """Generate high-level step-back question and conceptual rationale.

        Args:
            query: User's specific, low-level, or troubleshooting question.

        Returns:
            Tuple of (step_back_query, rationale).
        """
        logger.info("Generating Step-Back question for query: '%s'", query)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=STEP_BACK_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=STEP_BACK_USER_PROMPT.format(query=query)),
        ]

        resp = await self.llm.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=256,
        )

        content = resp.content.strip()

        # Parse structured output
        sb_match = re.search(r"Step-Back Question:\s*(.+)", content, re.IGNORECASE)
        rat_match = re.search(r"Rationale:\s*(.+)", content, re.IGNORECASE)

        if sb_match:
            step_back_query = sb_match.group(1).strip()
            rationale = rat_match.group(1).strip() if rat_match else None
        else:
            # Fallback: take first non-empty line
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            step_back_query = lines[0] if lines else query
            rationale = None

        logger.info("Step-Back question generated: '%s'", step_back_query)
        return step_back_query, rationale

    async def transform(self, query: str, **kwargs: Any) -> list[str]:
        """Transform specific query into its high-level step-back formulation."""
        step_back_q, _ = await self.generate_step_back(query)
        return [step_back_q]

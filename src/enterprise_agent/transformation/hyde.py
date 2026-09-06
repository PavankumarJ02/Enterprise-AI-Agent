"""Hypothetical Document Embeddings (HyDE) query transformer implementation."""

from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.transformation.base import QueryTransformer
from enterprise_agent.transformation.prompts import HYDE_SYSTEM_PROMPT, HYDE_USER_PROMPT

logger = get_logger(__name__)


class HyDETransformer(QueryTransformer):
    """Generates synthetic hypothetical answer passages to align vector query space."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def transform(self, query: str, num_hypotheses: int = 1, **kwargs: Any) -> list[str]:
        """Generate one or more hypothetical answer documents for the given query.

        Args:
            query: User's natural language question.
            num_hypotheses: Number of distinct hypothetical passages to synthesize.

        Returns:
            List of generated synthetic document passages.
        """
        logger.info("Generating HyDE hypothetical document for query: '%s'", query)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=HYDE_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=HYDE_USER_PROMPT.format(query=query)),
        ]

        hypotheses: list[str] = []
        for i in range(max(1, num_hypotheses)):
            # Slightly vary temperature for diversity if multiple hypotheses are requested
            temp = 0.2 + (i * 0.15) if num_hypotheses > 1 else 0.2
            resp = await self.llm.generate(
                messages=messages,
                temperature=temp,
                max_tokens=256,
            )
            clean_content = resp.content.strip().strip('"').strip("'")
            if clean_content:
                hypotheses.append(clean_content)

        if not hypotheses:
            logger.warning("HyDE generation produced empty output; falling back to original query")
            return [query]

        return hypotheses

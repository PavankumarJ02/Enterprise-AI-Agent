"""Multi-Query Expansion & Perspective Decomposition transformer implementation."""

import re
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.transformation.base import QueryTransformer
from enterprise_agent.transformation.prompts import (
    MULTI_QUERY_SYSTEM_PROMPT,
    MULTI_QUERY_USER_PROMPT,
)

logger = get_logger(__name__)


class MultiQueryTransformer(QueryTransformer):
    """Decomposes a query into multiple perspectives to overcome lexical mismatch."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def transform(self, query: str, num_variations: int = 3, **kwargs: Any) -> list[str]:
        """Generate multiple semantic reformulations and perspectives of the original query.

        Args:
            query: Original user query.
            num_variations: Desired number of alternative query variations.

        Returns:
            List of unique query reformulations including the original query.
        """
        logger.info("Generating %d Multi-Query variations for: '%s'", num_variations, query)

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=MULTI_QUERY_SYSTEM_PROMPT.format(num_variations=num_variations),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=MULTI_QUERY_USER_PROMPT.format(query=query),
            ),
        ]

        resp = await self.llm.generate(
            messages=messages,
            temperature=0.5,
            max_tokens=256,
        )

        variations: list[str] = [query]  # Always preserve original query as primary candidate
        seen: set[str] = {query.lower().strip()}

        for raw_line in resp.content.splitlines():
            # Strip numbering, bullets, quotes, and whitespace
            line = re.sub(r"^(?:\d+[\.\)]|\-|\*)\s*", "", raw_line.strip())
            line = line.strip('"').strip("'").strip()
            if not line:
                continue

            normalized = line.lower()
            if normalized not in seen:
                seen.add(normalized)
                variations.append(line)

            if len(variations) >= num_variations + 1:
                break

        logger.info(
            "Multi-query expansion generated %d unique queries from '%s'",
            len(variations),
            query,
        )
        return variations

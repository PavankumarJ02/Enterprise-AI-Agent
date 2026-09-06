"""Zero-shot LLM query intent classifier for ambiguous and edge-case queries."""

import json
import re

from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import ChatMessage, LLMProvider, MessageRole
from enterprise_agent.schemas.router import QueryIntent

logger = get_logger(__name__)

_ROUTER_SYSTEM_PROMPT = """You are an enterprise query routing classifier.
Analyze the user's query and categorize it into EXACTLY ONE of the following four intents:

1. direct_chat:
   - General conversation, chitchat, greetings, programming assistance,
     or general knowledge without internal company data.

2. rag_search:
   - Queries seeking qualitative information from enterprise handbooks,
     policies, guidelines, code of conduct, or unstructured documentation.

3. sql_database:
   - Structured questions asking for numerical facts, counts, sums,
     averages, employee details, departments, inventory, or sales orders.

4. autonomous_agent:
   - Compound or multi-step reasoning tasks that require combining data
     lookup with mathematical calculation, external clock telemetry,
     or multi-phase workflows.

You MUST respond ONLY with a valid JSON object in the following format, with no other text:
{
  "intent": "direct_chat" | "rag_search" | "sql_database" | "autonomous_agent",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation in one sentence>"
}
"""


class LLMRouter:
    """Zero-shot LLM intent classifier returning structured JSON classification decisions."""

    def __init__(self, llm: LLMProvider, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature

    async def classify(self, query: str) -> tuple[QueryIntent, float, str]:
        """Classify query using LLM reasoning and parse structured JSON result."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_ROUTER_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=f"Query to classify: {query}"),
        ]

        resp = await self.llm.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
        )
        raw_text = resp.content.strip()

        # Clean markdown fences
        clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        clean_json = re.sub(r"\s*```$", "", clean_json)

        try:
            parsed = json.loads(clean_json)
            intent_str = str(parsed.get("intent", "")).strip().lower()
            confidence = float(parsed.get("confidence", 0.85))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", "Classified by zero-shot LLM reasoning."))

            # Map to QueryIntent enum
            for intent_enum in QueryIntent:
                if intent_enum.value == intent_str:
                    return intent_enum, confidence, reasoning

            logger.warning("Unrecognized intent '%s' from LLM router; defaulting.", intent_str)
        except Exception as exc:
            logger.warning(
                "Failed to parse LLM router response as JSON: %s. Raw: %s", exc, raw_text
            )

        # Fallback regex search
        for intent_enum in QueryIntent:
            if re.search(rf"\b{intent_enum.value}\b", raw_text, re.IGNORECASE):
                return (
                    intent_enum,
                    0.75,
                    f"Parsed intent '{intent_enum.value}' via fallback text extraction.",
                )

        return QueryIntent.DIRECT_CHAT, 0.50, "Default fallback to direct chat."

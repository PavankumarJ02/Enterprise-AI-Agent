"""Prompt templates and message builders for grounded enterprise RAG synthesis."""

from enterprise_agent.llm.base import ChatMessage, MessageRole

DEFAULT_RAG_SYSTEM_PROMPT = """You are an expert Enterprise AI Knowledge Assistant.
Your responsibility is to provide accurate, professional, and directly grounded answers
based STRICTLY on the provided documentation context below.

CRITICAL RULES:
1. Rely ONLY on the facts explicitly mentioned in the provided Context.
2. Do NOT extrapolate, speculate, or introduce external knowledge not present in the Context.
3. If the Context does not contain enough information to fully and accurately answer the question,
   state clearly: "Based on the provided documentation, I do not have enough information to
   answer this question." Do not attempt to guess or synthesize an unverified answer.
4. Maintain an objective, concise, and professional tone.
5. Format your output using clean Markdown (bullet points, bold text, code blocks).
6. When stating any fact from Context, you MUST cite the source by appending its tag
   at the end of the sentence (e.g. "[Source 1]" or "[Source 1, 2]").
   Do NOT invent citations or cite sources not listed in the Context."""

USER_RAG_TEMPLATE = """Context Documentation:
---------------------
{context}
---------------------

User Question: {query}

Please provide a grounded answer based strictly on the context above."""


class RAGPromptBuilder:
    """Constructs grounded chat message sequences with anti-hallucination guardrails."""

    def __init__(self, default_system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT) -> None:
        self.default_system_prompt = default_system_prompt

    def build_messages(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
        history: list[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        """Assemble structured prompt with system guardrails and context blocks."""
        messages: list[ChatMessage] = []

        # 1. System Prompt
        sys_content = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else self.default_system_prompt
        )
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=sys_content))

        # 2. Conversational History (if provided)
        if history:
            messages.extend(history)

        # 3. Augmented User Turn
        user_content = USER_RAG_TEMPLATE.format(
            context=context.strip() if context.strip() else "[NO CONTEXT AVAILABLE]",
            query=query.strip(),
        )
        messages.append(ChatMessage(role=MessageRole.USER, content=user_content))

        return messages

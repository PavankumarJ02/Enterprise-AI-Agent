"""Deterministic enterprise mock fixtures, knowledge documents, and test harnesses."""

from collections.abc import AsyncIterator
from typing import Any

from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)
from enterprise_agent.sparse.base import SparseStore
from enterprise_agent.vectorstore.service import VectorSearchService

ENTERPRISE_FIXTURE_DOCUMENTS = [
    {
        "title": "Global HR Benefits & Stipends Policy",
        "content": (
            "# Global HR Benefits & Stipends Policy\n\n"
            "## Wellness and Equipment\n"
            "Full-time employees receive a $1,200 annual learning stipend for books, "
            "conferences, and verified courses. In addition, an annual health and wellness "
            "reimbursement of $600 is provided ($50 per month).\n\n"
            "## Parental Leave\n"
            "The company provides 16 weeks of fully paid parental leave for all new parents "
            "(birth, adoption, or foster placement) after 6 months of continuous service."
        ),
    },
    {
        "title": "Engineering Production Incident Runbook",
        "content": (
            "# Engineering Production Incident Runbook\n\n"
            "## Severity Classifications\n"
            "P1 incidents indicate critical production outages affecting more than 10% of users.\n"
            "The initial response SLA is 15 minutes. The incident commander must initiate a war\n"
            "room and page the on-call principal engineer immediately.\n\n"
            "## Rollback Procedures\n"
            "If a Kubernetes deployment exhibits error rates exceeding 1%, automated canary\n"
            "rollback triggers via ArgoCD within 90 seconds."
        ),
    },
    {
        "title": "SOC2 Type II Security Controls & Access Policy",
        "content": (
            "# SOC2 Type II Security Controls & Access Policy\n\n"
            "## Authentication Standards\n"
            "All internal corporate accounts must utilize hardware-backed FIDO2 Multi-Factor "
            "Authentication (MFA). Passwords must have a minimum length of 16 characters and "
            "be rotated every 90 days.\n\n"
            "## Audit Logging and Retention\n"
            "All production database queries, API access logs, and administrative changes must be "
            "retained in immutable WORM storage for 7 years."
        ),
    },
    {
        "title": "Corporate Travel & Expense Reimbursement Policy",
        "content": (
            "# Corporate Travel & Expense Reimbursement Policy\n\n"
            "## Daily Meal Per Diems\n"
            "Employees traveling on approved company business are entitled to a daily meal "
            "per diem of $75 for domestic travel and $100 for international travel.\n\n"
            "## Lodging Guidelines\n"
            "Standard hotel accommodations should not exceed $250 per night excluding taxes in "
            "Tier 1 metropolitan cities."
        ),
    },
]


async def seed_e2e_knowledge_base(
    ingestion_service: IngestionService,
    vector_service: VectorSearchService,
    sparse_store: SparseStore | None = None,
) -> list[str]:
    """Ingest and index all enterprise fixture documents into vector and sparse stores."""
    doc_ids = []
    for doc in ENTERPRISE_FIXTURE_DOCUMENTS:
        ingest_res = await ingestion_service.ingest_text(
            text=doc["content"],
            title=doc["title"],
            source=doc["title"],
        )
        doc_ids.append(ingest_res.document.id)

        # Index into vector store
        await vector_service.index_chunks(ingest_res.chunks)

        # Index into sparse store if provided
        if sparse_store:
            await sparse_store.index_chunks(ingest_res.chunks)

    return doc_ids


class DeterministicE2ELLM(LLMProvider):
    """Deterministic Mock LLM returning accurate enterprise answers based on keywords."""

    def __init__(self, model_name: str = "mock-enterprise-model") -> None:
        self.model_name = model_name

    def _resolve_response(self, messages: list[ChatMessage]) -> str:
        user_query = messages[-1].content.lower() if messages else ""

        # Check for specific enterprise queries
        if "parental leave" in user_query:
            return (
                "The company provides 16 weeks of fully paid parental leave for all new parents "
                "[Source 1]."
            )
        if "learning stipend" in user_query or "stipend" in user_query:
            return (
                "Employees receive a $1,200 annual learning stipend and $600 for wellness "
                "[Source 1]."
            )
        if "p1" in user_query or "incident" in user_query:
            return "P1 incidents have an initial response SLA of 15 minutes [Source 1]."
        if "soc2" in user_query or "password" in user_query or "retention" in user_query:
            return (
                "Passwords must be at least 16 characters and logs are retained for 7 years "
                "[Source 1]."
            )
        if "per diem" in user_query or "travel" in user_query:
            return (
                "Daily meal per diems are $75 for domestic and $100 for international travel "
                "[Source 1]."
            )
        if "action:" in user_query or "thought:" in user_query:
            # ReAct Loop mock response
            return "Thought: I know the final answer.\nFinal Answer: Task completed successfully."

        snippet = messages[-1].content if messages else ""
        return f"Standard enterprise response for query: {snippet}"

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        content = self._resolve_response(messages)
        return LLMResponse(
            content=content,
            model=self.model_name,
            usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        content = self._resolve_response(messages)
        words = content.split(" ")
        for i, word in enumerate(words):
            yield LLMStreamChunk(
                delta_text=word + (" " if i < len(words) - 1 else ""),
                finish_reason=None,
            )
        yield LLMStreamChunk(delta_text="", finish_reason="stop")

    async def health_check(self) -> bool:
        """Deterministic mock provider is always available."""
        return True

"""Knowledge base search tool wrapping hybrid retrieval and cross-encoder reranking."""

from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import RerankRequest
from enterprise_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class KnowledgeSearchTool(BaseTool):
    """Allows the agent to retrieve verified excerpts from the enterprise knowledge base."""

    def __init__(self, two_stage_service: TwoStageRetrievalService) -> None:
        self.two_stage_service = two_stage_service

    @property
    def name(self) -> str:
        """Tool name."""
        return "knowledge_search"

    @property
    def description(self) -> str:
        """Tool purpose and instructions."""
        return (
            "Search enterprise knowledge base for policies, technical documentation, runbooks, "
            "and guidelines. Input must be a specific search query. "
            "Returns ranked excerpts with source filenames and relevance scores."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema definition for parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or keyword query to search.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top excerpts to return (default: 3, max: 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute knowledge search and format excerpts."""
        query = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", 3))
        top_k = max(1, min(top_k, 10))

        if not query:
            return ToolResult(
                output="Error: No query provided for knowledge search.",
                is_error=True,
            )

        logger.info("Executing KnowledgeSearchTool: '%s' (top_k=%d)", query, top_k)
        try:
            req = RerankRequest(
                query=query,
                top_k=top_k,
                candidate_k=max(top_k * 4, 15),
                retrieval_strategy="hybrid",
            )
            response = await self.two_stage_service.retrieve_and_rerank(req)

            if not response.results:
                return ToolResult(
                    output=f"No relevant enterprise documentation found matching query: '{query}'.",
                    is_error=False,
                    metadata={"query": query, "total_results": 0},
                )

            formatted_excerpts: list[str] = []
            for rank, item in enumerate(response.results, start=1):
                source = item.metadata.get("source", "unknown")
                score_str = f"{item.rerank_score:.4f}"
                formatted_excerpts.append(
                    f"[{rank}] Source: {source} (Score: {score_str})\n{item.content.strip()}"
                )

            output_text = "\n\n".join(formatted_excerpts)
            return ToolResult(
                output=output_text,
                is_error=False,
                metadata={"query": query, "total_results": len(response.results)},
            )
        except Exception as exc:
            logger.error("Error executing KnowledgeSearchTool: %s", exc)
            return ToolResult(output=f"Knowledge search failed: {exc}", is_error=True)

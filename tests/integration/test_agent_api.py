"""Integration tests for Autonomous Agent API endpoints (/api/v1/agent)."""

from collections.abc import AsyncIterator, Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.agent.service import AgentService
from enterprise_agent.api.deps import (
    get_agent_service,
    get_embeddings,
    get_hybrid_search_service,
    get_ingestion_service,
    get_llm,
    get_reranker,
    get_sparse_store,
    get_tool_registry,
    get_two_stage_retrieval_service,
    get_vector_search_service,
    get_vector_store,
    reset_reranker,
    reset_sparse_store,
    reset_vector_store,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)
from enterprise_agent.main import create_application
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.clock import CurrentTimeTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.tools.search import KnowledgeSearchTool
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService


class ProgrammableMockLLM(LLMProvider):
    """Mock LLM that returns a pre-programmed sequence of responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        idx = min(self.call_count, len(self.responses) - 1)
        resp_text = self.responses[idx]
        self.call_count += 1
        return LLMResponse(
            content=resp_text,
            model="programmable-mock",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(delta_text="")

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def agent_client() -> Generator[TestClient, None, None]:
    """Test client configured with programmable LLM and registered tools."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_agent_col",
        sparse_search_provider="bm25",
        reranker_provider="mock",
    )
    app = create_application(settings)

    # Programmed mock responses: Call 1: Action, Call 2: Final Answer
    mock_llm = ProgrammableMockLLM(
        responses=[
            (
                "Thought: Calculating 50 * 4.\n"
                "Action: calculator\n"
                'Action Input: {"expression": "50 * 4"}'
            ),
            "Thought: I have the result.\nFinal Answer: The total is 200.",
        ]
    )

    ingestion_service = IngestionService()
    embeddings_service = get_embeddings_service(settings)
    vector_store = create_vector_store(settings)
    sparse_store = InMemoryBM25Store()
    mock_reranker = MockReranker()

    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
    )
    two_stage_service = TwoStageRetrievalService(
        hybrid_service=hybrid_service,
        vector_service=vector_service,
        reranker=mock_reranker,
    )

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(CurrentTimeTool())
    tool_registry.register(KnowledgeSearchTool(two_stage_service=two_stage_service))

    agent_service = AgentService(
        llm=mock_llm,
        tool_registry=tool_registry,
        default_max_iterations=5,
    )

    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_sparse_store] = lambda: sparse_store
    app.dependency_overrides[get_vector_search_service] = lambda: vector_service
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid_service
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_two_stage_retrieval_service] = lambda: two_stage_service
    app.dependency_overrides[get_tool_registry] = lambda: tool_registry
    app.dependency_overrides[get_agent_service] = lambda: agent_service

    reset_vector_store()
    reset_sparse_store()
    reset_reranker()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_list_tools_api(agent_client: TestClient) -> None:
    """Verify GET /api/v1/agent/tools lists registered tools and their parameters."""
    response = agent_client.get("/api/v1/agent/tools")
    assert response.status_code == 200
    data = response.json()

    assert len(data) >= 3
    tool_names = [t["name"] for t in data]
    assert "calculator" in tool_names
    assert "current_time" in tool_names
    assert "knowledge_search" in tool_names

    calc_tool = next(t for t in data if t["name"] == "calculator")
    assert "expression" in calc_tool["parameters"]["properties"]


def test_agent_chat_api_multi_turn(agent_client: TestClient) -> None:
    """Verify POST /api/v1/agent/chat executes tool and returns final answer with steps."""
    response = agent_client.post(
        "/api/v1/agent/chat",
        json={"query": "Calculate 50 * 4", "max_iterations": 4},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "Calculate 50 * 4"
    assert data["final_answer"] == "The total is 200."
    assert data["status"] == "success"
    assert data["iterations"] == 1
    assert len(data["steps"]) == 1

    step1 = data["steps"][0]
    assert step1["step_number"] == 1
    assert step1["action"] == "calculator"
    assert step1["action_input"] == {"expression": "50 * 4"}
    assert step1["observation"] == "200"


def test_agent_stream_api(agent_client: TestClient) -> None:
    """Verify POST /api/v1/agent/stream yields SSE events for thoughts, tools, and answer."""
    response = agent_client.post(
        "/api/v1/agent/stream",
        json={"query": "Calculate 50 * 4"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "event: thought" in content
    assert "event: tool_call" in content
    assert "event: tool_result" in content
    assert "event: final_answer" in content
    assert "data: [DONE]" in content


def test_agent_validation_errors(agent_client: TestClient) -> None:
    """Verify Pydantic validation rejects empty queries and negative iterations."""
    res_empty = agent_client.post("/api/v1/agent/chat", json={"query": ""})
    assert res_empty.status_code == 422

    res_invalid_iter = agent_client.post(
        "/api/v1/agent/chat",
        json={"query": "hello", "max_iterations": -5},
    )
    assert res_invalid_iter.status_code == 422

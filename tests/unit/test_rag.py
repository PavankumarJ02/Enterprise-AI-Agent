"""Unit tests for RAG prompt engineering, context assembly, and RAGService."""

from collections.abc import AsyncGenerator

import pytest
from qdrant_client import AsyncQdrantClient

from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.llm.base import ChatMessage, MessageRole
from enterprise_agent.llm.mock_client import MockLLMProvider
from enterprise_agent.rag.context import RAGContextAssembler, estimate_tokens
from enterprise_agent.rag.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    RAGPromptBuilder,
)
from enterprise_agent.rag.service import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    RAGService,
)
from enterprise_agent.schemas.rag import RAGQueryRequest
from enterprise_agent.schemas.search import SearchResultItem
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService


def test_estimate_tokens() -> None:
    """Verify estimate_tokens returns expected heuristic values."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens("a" * 100) == 25


def test_rag_prompt_builder() -> None:
    """Verify RAGPromptBuilder constructs system instructions and augmented context."""
    builder = RAGPromptBuilder()

    messages = builder.build_messages(
        query="What is the refund deadline?",
        context="[Source 1]\nRefunds must be requested within 30 days.",
    )

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == DEFAULT_RAG_SYSTEM_PROMPT
    assert messages[1].role == MessageRole.USER
    assert "Refunds must be requested within 30 days." in messages[1].content
    assert "What is the refund deadline?" in messages[1].content


def test_rag_prompt_builder_with_custom_system_and_history() -> None:
    """Verify custom system prompt and conversational history injection."""
    builder = RAGPromptBuilder()
    custom_sys = "Custom enterprise instructions."
    history = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi, how can I help?"),
    ]

    messages = builder.build_messages(
        query="Tell me more",
        context="Context details",
        system_prompt=custom_sys,
        history=history,
    )

    assert len(messages) == 4
    assert messages[0].content == custom_sys
    assert messages[1].content == "Hello"
    assert messages[2].content == "Hi, how can I help?"
    assert "Tell me more" in messages[3].content


def test_rag_context_assembler_formatting_and_sources() -> None:
    """Verify chunk block formatting with metadata labels and source extraction."""
    assembler = RAGContextAssembler()

    items = [
        SearchResultItem(
            chunk_id="c1",
            document_id="doc_hr",
            content="Employees get 25 days PTO.",
            score=0.95,
            chunk_index=0,
            metadata={"source": "handbook.pdf", "page": 12, "section": "Leave"},
        ),
        SearchResultItem(
            chunk_id="c2",
            document_id="doc_fin",
            content="Expenses must have receipts.",
            score=0.88,
            chunk_index=1,
            metadata={"source": "expenses.pdf", "page": 4},
        ),
    ]

    context, sources = assembler.assemble(items, max_context_tokens=1000)

    assert "Document: doc_hr" in context
    assert "File: handbook.pdf" in context
    assert "Section: Leave" in context
    assert "Employees get 25 days PTO." in context
    assert "Document: doc_fin" in context
    assert "Expenses must have receipts." in context

    assert len(sources) == 2
    assert sources[0].chunk_id == "c1"
    assert sources[0].score == 0.95
    assert sources[1].chunk_id == "c2"


def test_rag_context_assembler_budget_truncation() -> None:
    """Verify token ceiling limits accumulated chunks."""
    assembler = RAGContextAssembler()

    items = [
        SearchResultItem(
            chunk_id="c1",
            document_id="doc1",
            content="Alpha content " * 10,
            score=0.99,
            chunk_index=0,
            metadata={"source": "doc1.txt"},
        ),
        SearchResultItem(
            chunk_id="c2",
            document_id="doc2",
            content="Beta content " * 10,
            score=0.95,
            chunk_index=0,
            metadata={"source": "doc2.txt"},
        ),
    ]

    # Set very small token budget allowing only first chunk
    context, sources = assembler.assemble(items, max_context_tokens=60)

    assert len(sources) == 1
    assert sources[0].chunk_id == "c1"
    assert "Beta content" not in context


@pytest.fixture
async def rag_service_env() -> AsyncGenerator[tuple[RAGService, AsyncQdrantClient], None]:
    """Provide initialized RAGService connected to in-memory Qdrant and mock components."""
    client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="test_rag_col")
    provider = MockEmbeddingProvider(dimensions=8)
    embeddings_service = EmbeddingsService(provider=provider)
    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    llm = MockLLMProvider()
    service = RAGService(vector_service=vector_service, llm=llm)

    yield service, client
    await client.close()


@pytest.mark.asyncio
async def test_rag_service_query_grounded_answer(
    rag_service_env: tuple[RAGService, AsyncQdrantClient],
) -> None:
    """Verify end-to-end RAG query synthesis with indexed chunks."""
    service, _ = rag_service_env

    # 1. Index sample chunks
    chunks = [
        DocumentChunk(
            id="policy:chunk_0",
            document_id="policy",
            content="Remote work equipment is subsidized up to 500 dollars per calendar year.",
            chunk_index=0,
            metadata={"source": "remote_policy.md"},
        )
    ]
    await service.vector_service.index_chunks(chunks)

    # 2. Execute RAG query
    req = RAGQueryRequest(
        query="What is the remote work equipment subsidy amount?",
        top_k=3,
    )
    response = await service.query(req)

    assert response.query == req.query
    assert response.status == "success"
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "policy"
    assert response.model == "mock-enterprise-model"
    assert response.latency_ms > 0
    assert "[MOCK RESPONSE]" in response.answer
    assert response.grounding is not None


@pytest.mark.asyncio
async def test_rag_service_short_circuits_on_empty(
    rag_service_env: tuple[RAGService, AsyncQdrantClient],
) -> None:
    """Verify RAG query short-circuits gracefully when no documents match."""
    service, _ = rag_service_env

    req = RAGQueryRequest(
        query="What is the quantum teleportation policy?",
        top_k=5,
    )
    response = await service.query(req)

    assert response.status == "insufficient_context"
    assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(response.sources) == 0
    assert response.usage.total_tokens == 0
    assert response.grounding is not None
    assert response.grounding.grounding_status == "insufficient_context"


@pytest.mark.asyncio
async def test_rag_service_streaming(
    rag_service_env: tuple[RAGService, AsyncQdrantClient],
) -> None:
    """Verify query_stream emits sources, streams answer tokens, and emits grounding event."""
    service, _ = rag_service_env

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_stream",
            content="Streaming test content.",
            chunk_index=0,
            metadata={"source": "stream.txt"},
        )
    ]
    await service.vector_service.index_chunks(chunks)

    req = RAGQueryRequest(query="Test stream query")
    events: list[str] = []
    async for event in service.query_stream(req):
        events.append(event)

    assert len(events) >= 4
    # First event must be sources
    assert events[0].startswith("event: sources\n")
    assert "doc_stream" in events[0]
    # Penultimate event must be grounding
    grounding_events = [e for e in events if e.startswith("event: grounding\n")]
    assert len(grounding_events) == 1
    # Last event must be DONE
    assert events[-1] == "data: [DONE]\n\n"

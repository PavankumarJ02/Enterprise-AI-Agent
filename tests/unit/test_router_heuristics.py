"""Unit tests for HeuristicRouter."""

import pytest

from enterprise_agent.router.heuristics import HeuristicRouter
from enterprise_agent.schemas.router import QueryIntent


@pytest.fixture
def heuristic_router() -> HeuristicRouter:
    """Fixture providing an instance of HeuristicRouter."""
    return HeuristicRouter()


@pytest.mark.parametrize(
    "greeting",
    [
        "hello",
        "Hello!",
        "hi",
        "hey there",
        "good morning",
        "good evening",
        "how are you?",
        "thanks a lot",
        "thank you",
        "bye",
    ],
)
def test_heuristic_classifies_greetings_as_direct_chat(
    heuristic_router: HeuristicRouter,
    greeting: str,
) -> None:
    """Verify conversational greetings and acknowledgments are routed to DIRECT_CHAT."""
    result = heuristic_router.classify(greeting)
    assert result is not None
    intent, conf, _ = result
    assert intent == QueryIntent.DIRECT_CHAT
    assert conf >= 0.95


@pytest.mark.parametrize(
    "compound_query",
    [
        "Look up the travel policy and then calculate total meal cost for 4 employees",
        "Find the server price in catalog and compute the 15% discount",
        "Calculate the quarterly bonus and draft a summary report for HR",
        "Check the current time and verify if Asian offices are open",
    ],
)
def test_heuristic_classifies_compound_actions_as_agent(
    heuristic_router: HeuristicRouter,
    compound_query: str,
) -> None:
    """Verify compound tasks combining lookup and calculation are routed to AUTONOMOUS_AGENT."""
    result = heuristic_router.classify(compound_query)
    assert result is not None
    intent, conf, _ = result
    assert intent == QueryIntent.AUTONOMOUS_AGENT
    assert conf >= 0.90


@pytest.mark.parametrize(
    "sql_query",
    [
        "How many employees work in Engineering?",
        "Count of products in stock",
        "What is the average salary of senior architects?",
        "What is the budget for Global Marketing?",
        "List all departments in Building A",
        "SELECT * FROM products WHERE price > 5000",
    ],
)
def test_heuristic_classifies_metrics_as_sql(
    heuristic_router: HeuristicRouter,
    sql_query: str,
) -> None:
    """Verify structured database metrics queries are routed to SQL_DATABASE."""
    result = heuristic_router.classify(sql_query)
    assert result is not None
    intent, conf, _ = result
    assert intent == QueryIntent.SQL_DATABASE
    assert conf >= 0.90


@pytest.mark.parametrize(
    "policy_query",
    [
        "What is the company parental leave policy?",
        "Where can I find the employee code of conduct documentation?",
        "What are the guidelines for remote work equipment reimbursement?",
        "What is the rule regarding unused vacation rollover?",
        "How do I submit an annual performance review self-evaluation?",
    ],
)
def test_heuristic_classifies_handbook_as_rag(
    heuristic_router: HeuristicRouter,
    policy_query: str,
) -> None:
    """Verify documentation and policy questions are routed to RAG_SEARCH."""
    result = heuristic_router.classify(policy_query)
    assert result is not None
    intent, conf, _ = result
    assert intent == QueryIntent.RAG_SEARCH
    assert conf >= 0.85


def test_heuristic_returns_none_on_ambiguous_queries(
    heuristic_router: HeuristicRouter,
) -> None:
    """Verify ambiguous, subjective, or border-case queries return None for cascade to handle."""
    assert heuristic_router.classify("Can you give me advice on leadership?") is None
    assert heuristic_router.classify("Explain quantum computing concepts.") is None
    assert heuristic_router.classify("   ") is None

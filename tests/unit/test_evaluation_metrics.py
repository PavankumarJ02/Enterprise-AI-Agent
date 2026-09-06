"""Unit tests for RAG evaluation metrics calculation."""

import pytest

from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.evaluation.metrics import (
    calculate_answer_relevance,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
)


def test_faithfulness_grounded_answer() -> None:
    """Verify high faithfulness score for fully grounded answers."""
    contexts = [
        "Employees receive up to $100 monthly internet stipend and "
        "$500 home office equipment credit."
    ]
    answer = "The monthly internet allowance is $100 and equipment reimbursement is $500."
    score = calculate_faithfulness(answer, contexts)
    assert score >= 0.90


def test_faithfulness_hallucinated_answer() -> None:
    """Verify lower faithfulness score for unsupported fabricated claims."""
    contexts = [
        "Employees receive up to $100 monthly internet stipend and "
        "$500 home office equipment credit."
    ]
    answer = (
        "The monthly internet allowance is $900 and employees also get a free company Ferrari car. "
        "They can work from Mars on Fridays."
    )
    score = calculate_faithfulness(answer, contexts)
    assert score <= 0.40


def test_faithfulness_insufficient_information_safe() -> None:
    """Verify that honest refusal of insufficient context is counted as faithful (not penalized)."""
    contexts = ["The document discusses company history from 2010."]
    answer = "I do not have enough information in the provided context to answer your question."
    score = calculate_faithfulness(answer, contexts)
    assert score == 1.0


@pytest.mark.asyncio
async def test_answer_relevance_lexical() -> None:
    """Verify lexical answer relevance distinguishes matching from unrelated answers."""
    query = "What is the policy for booking flights?"
    relevant_ans = (
        "For booking flights, economy is required under 6 hours and business over 6 hours."
    )
    irrelevant_ans = "The cafeteria serves pizza and tacos every Tuesday afternoon."

    rel_score = await calculate_answer_relevance(query, relevant_ans, embedding_provider=None)
    irrel_score = await calculate_answer_relevance(query, irrelevant_ans, embedding_provider=None)

    assert rel_score >= 0.50
    assert irrel_score < rel_score


@pytest.mark.asyncio
async def test_answer_relevance_embedding() -> None:
    """Verify embedding-based answer relevance calculation with MockEmbeddingProvider."""
    provider = MockEmbeddingProvider(dimensions=16)
    query = "What is the laptop replacement schedule?"
    answer = "Engineering laptops are replaced every 3 years."

    score = await calculate_answer_relevance(query, answer, embedding_provider=provider)
    assert 0.0 <= score <= 1.0


def test_context_precision_perfect_ranking() -> None:
    """Verify high precision when relevant context is ranked at position 1."""
    ground_truth = "Hotel lodging reimbursement is capped at $250 per night in Tier 1 cities."
    contexts = [
        "Section 4.2: Lodging expenses are capped at $250 per night in Tier 1 cities.",
        "Section 1.1: General workplace guidelines and dress code rules.",
        "Section 9.0: Parking permit applications and monthly garage fees.",
    ]
    precision = calculate_context_precision(contexts, ground_truth)
    assert precision == 1.0


def test_context_precision_lower_for_delayed_ranking() -> None:
    """Verify lower precision when relevant context appears later in retrieved list."""
    ground_truth = "Hotel lodging reimbursement is capped at $250 per night in Tier 1 cities."
    contexts = [
        "Section 1.1: General workplace guidelines and dress code rules.",
        "Section 9.0: Parking permit applications and monthly garage fees.",
        "Section 4.2: Lodging expenses are capped at $250 per night in Tier 1 cities.",
    ]
    precision = calculate_context_precision(contexts, ground_truth)
    # Relevant chunk at index 3 -> Precision@3 = 1/3 = 0.3333
    assert precision <= 0.40


def test_context_recall_full_coverage() -> None:
    """Verify full context recall when all ground truth facts exist in context."""
    ground_truth = (
        "Remote staff get a $100 internet stipend. "
        "They can also expense $500 for ergonomic home equipment."
    )
    contexts = [
        "Section 3.1: Remote employees receive a $100 monthly internet stipend.",
        "New staff are eligible for a $500 ergonomic equipment reimbursement upon onboarding.",
    ]
    recall = calculate_context_recall(contexts, ground_truth)
    assert recall == 1.0


def test_context_recall_partial_coverage() -> None:
    """Verify partial recall when context only contains a subset of ground truth facts."""
    ground_truth = (
        "Remote staff get a $100 internet stipend. "
        "They can also expense $500 for ergonomic home equipment."
    )
    contexts = ["Section 3.1: Remote employees receive a $100 monthly internet stipend."]
    recall = calculate_context_recall(contexts, ground_truth)
    assert recall == 0.50

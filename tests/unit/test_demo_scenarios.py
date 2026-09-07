"""Unit tests for interactive demonstration suite and scenario handlers."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.demo_scenarios import (
    SCENARIO_REGISTRY,
    DemoHarness,
    execute_all_scenarios,
    execute_scenario,
    run_scenario_1,
    run_scenario_2,
    run_scenario_3,
    run_scenario_4,
    run_scenario_5,
    run_scenario_6,
)


def test_scenario_registry_completeness() -> None:
    """Verify that all 6 expected enterprise scenarios are registered with valid metadata."""
    assert len(SCENARIO_REGISTRY) == 6
    for num in range(1, 7):
        assert num in SCENARIO_REGISTRY
        scen = SCENARIO_REGISTRY[num]
        assert scen.number == num
        assert scen.title.strip(), f"Scenario {num} must have a non-empty title."
        assert scen.description.strip(), f"Scenario {num} must have a description."
        assert callable(scen.handler), f"Scenario {num} handler must be callable."


@pytest.fixture
def demo_harness() -> Generator[DemoHarness, None, None]:
    """Provide a clean, seeded DemoHarness in a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        harness = DemoHarness(temp_dir=Path(temp_dir))
        yield harness


@pytest.mark.asyncio
async def test_scenario_1_rag(demo_harness: DemoHarness) -> None:
    """Verify Scenario 1: Grounded RAG with citations."""
    await demo_harness.initialize()
    result = await run_scenario_1(demo_harness)

    assert result["status"] == "success"
    assert result["answer"].strip()
    assert result["sources_count"] >= 1
    assert result["grounding_status"] in ("verified", "partially_grounded", "unverified")


@pytest.mark.asyncio
async def test_scenario_2_router(demo_harness: DemoHarness) -> None:
    """Verify Scenario 2: Semantic Router cascade decisions."""
    await demo_harness.initialize()
    result = await run_scenario_2(demo_harness)

    assert result["status"] == "success"
    assert len(result["decisions"]) == 4
    for dec in result["decisions"]:
        assert dec["intent"] in (
            "direct_chat",
            "rag_search",
            "sql_database",
            "autonomous_agent",
        )
        assert 0.0 <= dec["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_scenario_3_sql(demo_harness: DemoHarness) -> None:
    """Verify Scenario 3: Safe SQL execution and mutation interception."""
    await demo_harness.initialize()
    result = await run_scenario_3(demo_harness)

    assert result["status"] == "success"
    assert result["safe_query_rows"] > 0
    assert result["mutation_blocked"] is True


@pytest.mark.asyncio
async def test_scenario_4_guardrails(demo_harness: DemoHarness) -> None:
    """Verify Scenario 4: Guardrails prompt-injection blocking and PII masking."""
    result = await run_scenario_4(demo_harness)

    assert result["status"] == "success"
    assert result["jailbreak_blocked"] is True
    assert result["pii_redacted_count"] >= 2


@pytest.mark.asyncio
async def test_scenario_5_agent(demo_harness: DemoHarness) -> None:
    """Verify Scenario 5: ReAct autonomous decision agent execution."""
    result = await run_scenario_5(demo_harness)

    assert result["status"] == "success"
    assert result["steps_count"] >= 1
    assert result["final_answer"].strip()


@pytest.mark.asyncio
async def test_scenario_6_experiments(demo_harness: DemoHarness) -> None:
    """Verify Scenario 6: Experiment tracking and run delta comparison."""
    result = await run_scenario_6(demo_harness)

    assert result["status"] == "success"
    assert "baseline_dense_ann_k3" in result["baseline"]
    assert "candidate_hybrid_rerank_k5" in result["candidate"]
    assert "faithfulness" in result["metrics_compared"]


@pytest.mark.asyncio
async def test_execute_all_scenarios_pipeline() -> None:
    """Verify full sequential scenario execution pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        harness = DemoHarness(temp_dir=Path(temp_dir))
        await harness.initialize()

        results = await execute_all_scenarios(harness, delay_seconds=0.0)
        assert len(results) == 6
        for res in results:
            assert res["status"] == "success"


@pytest.mark.asyncio
async def test_execute_scenario_invalid_number(demo_harness: DemoHarness) -> None:
    """Verify invalid scenario numbers raise ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario number"):
        await execute_scenario(99, demo_harness)

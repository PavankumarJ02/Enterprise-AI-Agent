"""Unit tests for the enterprise evaluation benchmark dataset."""

from enterprise_agent.evaluation.dataset import ENTERPRISE_BENCHMARK_DATASET


def test_benchmark_dataset_integrity() -> None:
    """Verify dataset contains 10 samples with complete ground truth and golden contexts."""
    assert len(ENTERPRISE_BENCHMARK_DATASET) == 10

    queries: set[str] = set()
    for sample in ENTERPRISE_BENCHMARK_DATASET:
        assert sample.query.strip(), "Sample query cannot be empty"
        assert sample.ground_truth.strip(), "Ground truth cannot be empty"
        assert len(sample.golden_contexts) >= 1, "Must have at least one golden context passage"
        assert sample.golden_contexts[0].strip(), "Golden context passage cannot be empty"

        assert sample.query not in queries, f"Duplicate query detected: {sample.query}"
        queries.add(sample.query)

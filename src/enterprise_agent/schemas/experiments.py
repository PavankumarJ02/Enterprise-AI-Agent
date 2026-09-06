"""Schemas and data contracts for experiment tracking and benchmark comparisons."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRunCreate(BaseModel):
    """Request payload to create a new experiment run."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(
        default="rag_optimization",
        description="Name of the experiment grouping (e.g. 'rag_optimization').",
    )
    run_name: str = Field(..., description="Human-readable name for this specific trial run.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters, model choices, top-k, and retrieval strategy parameters.",
    )
    tags: list[str] = Field(
        default_factory=list, description="Categorical tags for filtering and grouping."
    )


class ExperimentRunUpdate(BaseModel):
    """Request payload to update metrics on an existing run."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float] = Field(
        default_factory=dict, description="Recorded metric scores (e.g. faithfulness, recall)."
    )
    status: str = Field(default="completed", description="Execution status of the run.")
    total_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Total execution duration in milliseconds."
    )


class ExperimentRun(BaseModel):
    """Full representation of a persisted experiment run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique UUID identifier for the run.")
    experiment_name: str = Field(..., description="Parent experiment group name.")
    run_name: str = Field(..., description="Descriptive trial run name.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Configuration hyperparameters."
    )
    metrics: dict[str, float] = Field(default_factory=dict, description="Calculated metric scores.")
    tags: list[str] = Field(default_factory=list, description="Categorical tags.")
    status: str = Field(
        default="completed", description="Status ('running', 'completed', 'failed')."
    )
    total_latency_ms: float = Field(default=0.0, description="Total wall-clock duration.")
    created_at: str = Field(..., description="ISO 8601 creation timestamp.")


class MetricDelta(BaseModel):
    """Comparative performance delta between candidate run and baseline."""

    model_config = ConfigDict(frozen=True)

    baseline_val: float = Field(..., description="Metric value from baseline run.")
    candidate_val: float = Field(..., description="Metric value from candidate run.")
    absolute_delta: float = Field(..., description="candidate_val - baseline_val.")
    percentage_delta: float = Field(
        ..., description="Percentage improvement or degradation relative to baseline."
    )
    improved: bool = Field(
        ..., description="True if candidate outperformed baseline on this metric."
    )


class ExperimentCompareRequest(BaseModel):
    """Request to perform side-by-side comparison across runs."""

    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(
        ..., min_length=2, description="List of at least two run IDs to compare."
    )
    baseline_run_id: str | None = Field(
        default=None,
        description="Optional run ID to designate as baseline (defaults to first run).",
    )


class RunComparison(BaseModel):
    """Comparative analysis matrix between baseline and candidate runs."""

    model_config = ConfigDict(frozen=True)

    baseline_run: ExperimentRun = Field(..., description="The baseline reference run.")
    candidate_runs: list[ExperimentRun] = Field(
        ..., description="Candidate runs compared against baseline."
    )
    metric_deltas: dict[str, dict[str, MetricDelta]] = Field(
        ...,
        description="Mapping of candidate_run_id -> metric_name -> MetricDelta comparison.",
    )
    best_runs_per_metric: dict[str, str] = Field(
        ..., description="Mapping of metric_name -> run_id of highest-scoring run."
    )


class GridSweepRequest(BaseModel):
    """Request to trigger an automated hyperparameter grid sweep."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(
        default="retrieval_grid_sweep",
        description="Name of the experiment to log grid runs under.",
    )
    retrieval_strategies: list[str] = Field(
        default=["dense", "hybrid", "hybrid_rerank"],
        description="List of retrieval strategies to test in the grid.",
    )
    top_k_values: list[int] = Field(
        default=[3, 5],
        description="List of top_k values to evaluate per strategy.",
    )


class GridSweepResponse(BaseModel):
    """Summary of executed hyperparameter grid sweep."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(..., description="Experiment group name.")
    total_runs: int = Field(..., ge=0, description="Total number of evaluated trial runs.")
    runs: list[ExperimentRun] = Field(..., description="List of all logged trial runs.")
    best_run_id: str = Field(..., description="Run ID achieving highest overall score.")
    best_metric: str = Field(..., description="Metric used to select best run.")

"""Schemas and data models for RAG quantitative evaluation and benchmarking."""

from pydantic import BaseModel, ConfigDict, Field


class EvaluationSample(BaseModel):
    """Single test sample for RAG evaluation."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="User query or test prompt.")
    ground_truth: str = Field(..., description="Golden factual reference answer.")
    golden_contexts: list[str] = Field(
        default_factory=list,
        description="Reference passages containing factual evidence for the query.",
    )


class SampleEvaluationResult(BaseModel):
    """Evaluation metrics computed for a single RAG query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="The evaluated user query.")
    answer: str = Field(..., description="Generated answer under evaluation.")
    retrieved_contexts: list[str] = Field(
        default_factory=list, description="Passages provided in the RAG prompt context."
    )
    ground_truth: str = Field(..., description="Ground truth reference answer.")
    faithfulness: float = Field(
        ..., ge=0.0, le=1.0, description="Faithfulness of answer claims to retrieved context."
    )
    answer_relevance: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic relevance of answer to user query."
    )
    context_precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision of retrieved contexts relative to ground truth."
    )
    context_recall: float = Field(
        ..., ge=0.0, le=1.0, description="Coverage of ground truth facts within retrieved contexts."
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Latency in milliseconds for executing the query."
    )


class MetricSummary(BaseModel):
    """Statistical summary and pass/fail determination for an evaluation metric."""

    model_config = ConfigDict(frozen=True)

    mean: float = Field(..., description="Mean score across all evaluated samples.")
    min: float = Field(..., description="Minimum score observed in the evaluation run.")
    max: float = Field(..., description="Maximum score observed in the evaluation run.")
    threshold: float = Field(..., description="Target threshold defined in system configuration.")
    passed: bool = Field(
        ..., description="True if mean score satisfies or exceeds configured threshold."
    )


class EvaluationReport(BaseModel):
    """Dataset-level evaluation report aggregating all sample scores and metrics."""

    model_config = ConfigDict(frozen=True)

    total_samples: int = Field(..., ge=0, description="Total number of evaluated samples.")
    metrics: dict[str, MetricSummary] = Field(
        ..., description="Summary statistics for faithfulness, relevance, precision, and recall."
    )
    overall_passed: bool = Field(
        ..., description="True if all four core RAG metrics satisfy their target thresholds."
    )
    samples: list[SampleEvaluationResult] = Field(
        default_factory=list, description="Individual evaluation results per sample."
    )
    total_latency_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock duration of the evaluation run."
    )


class RunEvaluationRequest(BaseModel):
    """Request payload to trigger an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str | None = Field(
        default="enterprise_benchmark",
        description="Named benchmark dataset to run, e.g. 'enterprise_benchmark'.",
    )
    samples: list[EvaluationSample] | None = Field(
        default=None,
        description="Custom evaluation samples to evaluate instead of named dataset.",
    )
    use_live_rag: bool = Field(
        default=False,
        description="If True, queries live RAGService to generate answer and contexts.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of passages to retrieve per query when running live RAG.",
    )


class BenchmarkDatasetResponse(BaseModel):
    """Response containing benchmark dataset questions and golden answers."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Name of the benchmark dataset.")
    total_samples: int = Field(..., ge=0, description="Total sample count in dataset.")
    samples: list[EvaluationSample] = Field(
        ..., description="List of evaluation samples with golden contexts."
    )

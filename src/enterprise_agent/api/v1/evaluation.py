"""REST API endpoints for RAG quantitative evaluation and benchmarking."""

from fastapi import APIRouter, Depends, status

from enterprise_agent.api.deps import get_evaluation_service
from enterprise_agent.evaluation.dataset import ENTERPRISE_BENCHMARK_DATASET
from enterprise_agent.evaluation.service import RAGEvaluationService
from enterprise_agent.schemas.evaluation import (
    BenchmarkDatasetResponse,
    EvaluationReport,
    RunEvaluationRequest,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarking"])


@router.post(
    "/run",
    response_model=EvaluationReport,
    status_code=status.HTTP_200_OK,
    summary="Run quantitative RAG evaluation",
    description=(
        "Executes Faithfulness, Answer Relevance, Context Precision, and Context Recall "
        "evaluations over a benchmark dataset or custom test samples."
    ),
)
async def run_evaluation(
    request: RunEvaluationRequest,
    service: RAGEvaluationService = Depends(get_evaluation_service),
) -> EvaluationReport:
    """Execute evaluation benchmark over requested dataset or samples."""
    samples = request.samples if request.samples is not None else ENTERPRISE_BENCHMARK_DATASET
    return await service.evaluate_dataset(
        samples=samples,
        use_live_rag=request.use_live_rag,
        top_k=request.top_k,
    )


@router.get(
    "/dataset",
    response_model=BenchmarkDatasetResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve default enterprise benchmark dataset",
    description="Returns the curated 10-sample golden reference dataset with golden contexts.",
)
async def get_benchmark_dataset() -> BenchmarkDatasetResponse:
    """Retrieve the curated enterprise benchmark dataset."""
    return BenchmarkDatasetResponse(
        name="enterprise_benchmark",
        total_samples=len(ENTERPRISE_BENCHMARK_DATASET),
        samples=ENTERPRISE_BENCHMARK_DATASET,
    )

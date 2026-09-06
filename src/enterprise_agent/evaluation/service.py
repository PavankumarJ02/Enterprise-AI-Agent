"""Service coordinating dataset-level and sample-level RAG evaluation."""

import time

from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.base import EmbeddingProvider
from enterprise_agent.evaluation.dataset import ENTERPRISE_BENCHMARK_DATASET
from enterprise_agent.evaluation.metrics import (
    calculate_answer_relevance,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
)
from enterprise_agent.rag.service import RAGService
from enterprise_agent.schemas.evaluation import (
    EvaluationReport,
    EvaluationSample,
    MetricSummary,
    SampleEvaluationResult,
)
from enterprise_agent.schemas.rag import RAGQueryRequest

logger = get_logger(__name__)


class RAGEvaluationService:
    """Quantitative evaluation engine calculating Faithfulness, Relevance, Precision, and Recall."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        rag_service: RAGService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.rag_service = rag_service
        self.settings = settings or get_settings()

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        answer: str | None = None,
        contexts: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> SampleEvaluationResult:
        """Evaluate a single query sample against reference golden data."""
        eval_answer = answer if answer is not None else sample.ground_truth
        eval_contexts = contexts if contexts is not None else sample.golden_contexts

        faithfulness = calculate_faithfulness(eval_answer, eval_contexts)
        relevance = await calculate_answer_relevance(
            sample.query, eval_answer, self.embedding_provider
        )
        precision = calculate_context_precision(eval_contexts, sample.ground_truth)
        recall = calculate_context_recall(eval_contexts, sample.ground_truth)

        return SampleEvaluationResult(
            query=sample.query,
            answer=eval_answer,
            retrieved_contexts=eval_contexts,
            ground_truth=sample.ground_truth,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            latency_ms=round(latency_ms, 2),
        )

    async def evaluate_dataset(
        self,
        samples: list[EvaluationSample] | None = None,
        use_live_rag: bool = False,
        top_k: int = 3,
    ) -> EvaluationReport:
        """Evaluate full dataset of samples and aggregate statistical summaries."""
        start_time = time.perf_counter()
        eval_samples = samples if samples is not None else ENTERPRISE_BENCHMARK_DATASET
        results: list[SampleEvaluationResult] = []

        logger.info(
            "Starting RAG evaluation over %d samples (live_rag=%s, top_k=%d)",
            len(eval_samples),
            use_live_rag,
            top_k,
        )

        for sample in eval_samples:
            if use_live_rag and self.rag_service:
                t0 = time.perf_counter()
                rag_req = RAGQueryRequest(query=sample.query, top_k=top_k)
                rag_resp = await self.rag_service.query(rag_req)
                sample_latency = (time.perf_counter() - t0) * 1000.0
                contexts = [src.content for src in rag_resp.sources]
                sample_res = await self.evaluate_sample(
                    sample=sample,
                    answer=rag_resp.answer,
                    contexts=contexts,
                    latency_ms=sample_latency,
                )
            else:
                sample_res = await self.evaluate_sample(sample=sample, latency_ms=0.0)
            results.append(sample_res)

        # Statistical summaries
        def _summarize(values: list[float], threshold: float) -> MetricSummary:
            if not values:
                return MetricSummary(mean=0.0, min=0.0, max=0.0, threshold=threshold, passed=False)
            mean_val = round(sum(values) / len(values), 4)
            return MetricSummary(
                mean=mean_val,
                min=round(min(values), 4),
                max=round(max(values), 4),
                threshold=threshold,
                passed=mean_val >= threshold,
            )

        metrics: dict[str, MetricSummary] = {
            "faithfulness": _summarize(
                [r.faithfulness for r in results],
                self.settings.eval_faithfulness_threshold,
            ),
            "answer_relevance": _summarize(
                [r.answer_relevance for r in results],
                self.settings.eval_answer_relevance_threshold,
            ),
            "context_precision": _summarize(
                [r.context_precision for r in results],
                self.settings.eval_context_precision_threshold,
            ),
            "context_recall": _summarize(
                [r.context_recall for r in results],
                self.settings.eval_context_recall_threshold,
            ),
        }

        overall_passed = all(m.passed for m in metrics.values())
        total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)

        return EvaluationReport(
            total_samples=len(results),
            metrics=metrics,
            overall_passed=overall_passed,
            samples=results,
            total_latency_ms=total_latency,
        )

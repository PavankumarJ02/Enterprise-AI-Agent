"""Performance and Latency Benchmarking CLI for Enterprise AI Agent.

Measures cold vs. warm caching latency, throughput (QPS), and concurrent tool dispatch speedups.
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import setup_logging
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.performance.cache import QueryEmbeddingCache, RetrievalCache
from enterprise_agent.reranking.factory import create_reranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import RerankRequest
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.clock import CurrentTimeTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService
from tests.fixtures.enterprise_fixtures import seed_e2e_knowledge_base

# ANSI Color codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[1;36m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_BLUE = "\033[1;34m"


def calc_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Calculate P50, P90, P99, and mean from a list of millisecond latencies."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return round(sorted_lats[f] + (k - f) * (sorted_lats[c] - sorted_lats[f]), 2)

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "mean": round(statistics.mean(sorted_lats), 2),
    }


async def run_embedding_benchmarks(
    embeddings_service: EmbeddingsService,
    queries: list[str],
) -> dict[str, Any]:
    """Measure cold vs warm query embedding latency and speedup ratio."""
    # 1. Cold pass (cache miss)
    cold_times: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        await embeddings_service.embed_query(q)
        cold_times.append((time.perf_counter() - t0) * 1000)

    # 2. Warm pass (cache hit)
    warm_times: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        await embeddings_service.embed_query(q)
        warm_times.append((time.perf_counter() - t0) * 1000)

    cold_stats = calc_percentiles(cold_times)
    warm_stats = calc_percentiles(warm_times)
    speedup = round(cold_stats["mean"] / warm_stats["mean"], 2) if warm_stats["mean"] > 0 else 1.0

    return {
        "cold": cold_stats,
        "warm": warm_stats,
        "speedup_ratio": speedup,
        "cache_stats": embeddings_service.cache.get_stats() if embeddings_service.cache else {},
    }


async def run_retrieval_benchmarks(
    retrieval_service: TwoStageRetrievalService,
    queries: list[str],
) -> dict[str, Any]:
    """Measure cold vs warm hybrid retrieval and reranking latency."""
    # 1. Cold pass
    cold_times: list[float] = []
    for q in queries:
        req = RerankRequest(query=q, top_k=3, candidate_k=5)
        t0 = time.perf_counter()
        await retrieval_service.retrieve_and_rerank(req)
        cold_times.append((time.perf_counter() - t0) * 1000)

    # 2. Warm pass
    warm_times: list[float] = []
    for q in queries:
        req = RerankRequest(query=q, top_k=3, candidate_k=5)
        t0 = time.perf_counter()
        await retrieval_service.retrieve_and_rerank(req)
        warm_times.append((time.perf_counter() - t0) * 1000)

    cold_stats = calc_percentiles(cold_times)
    warm_stats = calc_percentiles(warm_times)
    speedup = round(cold_stats["mean"] / warm_stats["mean"], 2) if warm_stats["mean"] > 0 else 1.0

    return {
        "cold": cold_stats,
        "warm": warm_stats,
        "speedup_ratio": speedup,
        "cache_stats": retrieval_service.cache.get_stats() if retrieval_service.cache else {},
    }


async def run_concurrent_tools_benchmark(tool_registry: ToolRegistry) -> dict[str, Any]:
    """Compare sequential vs concurrent multi-tool execution latency."""
    tool_calls = [
        ("calculator", {"expression": "25 * 40 + 150"}),
        ("current_time", {}),
        ("calculator", {"expression": "1000 / 8 + 45"}),
        ("current_time", {}),
    ]

    # Sequential execution
    t0 = time.perf_counter()
    for name, args in tool_calls:
        await tool_registry.execute(name, args)
    seq_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Concurrent execution via execute_many
    t1 = time.perf_counter()
    await tool_registry.execute_many(tool_calls)
    conc_time_ms = round((time.perf_counter() - t1) * 1000, 2)

    speedup = round(seq_time_ms / conc_time_ms, 2) if conc_time_ms > 0 else 1.0

    return {
        "sequential_ms": seq_time_ms,
        "concurrent_ms": conc_time_ms,
        "speedup_ratio": speedup,
    }


async def main() -> None:
    """Entrypoint running comprehensive benchmark suite."""
    parser = argparse.ArgumentParser(description="Latency & Throughput Performance Benchmark")
    parser.add_argument("--iterations", type=int, default=15, help="Number of benchmark iterations")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock providers")
    args = parser.parse_args()

    setup_logging(level="WARNING")

    title = "Enterprise AI Agent - Performance & Latency Benchmark"
    print(f"\n{C_CYAN}{'=' * 76}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {title.center(72)}  {C_RESET}")
    print(f"{C_CYAN}{'=' * 76}{C_RESET}\n")

    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        performance_cache_enabled=True,
    )

    # Setup infrastructure
    emb_cache = QueryEmbeddingCache(max_size=500, ttl_seconds=3600)
    ret_cache = RetrievalCache(max_size=500, ttl_seconds=1800)

    embedding_provider = MockEmbeddingProvider(dimensions=16)
    embeddings_service = EmbeddingsService(provider=embedding_provider, cache=emb_cache)

    qdrant_client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="benchmark_vectors")
    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    sparse_store = InMemoryBM25Store()

    ingestion_service = IngestionService()
    await seed_e2e_knowledge_base(
        ingestion_service=ingestion_service,
        vector_service=vector_service,
        sparse_store=sparse_store,
    )

    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
    )
    reranker = create_reranker(settings)
    retrieval_service = TwoStageRetrievalService(
        hybrid_service=hybrid_service,
        vector_service=vector_service,
        reranker=reranker,
        cache=ret_cache,
    )

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(CurrentTimeTool())

    test_queries = [
        "What is the annual learning stipend for employees?",
        "What is the parental leave policy duration?",
        "What are the SOC2 audit log retention requirements?",
        "What is the daily meal per diem for domestic travel?",
        "How do I request a Kubernetes namespace deployment?",
    ] * (args.iterations // 5 + 1)
    test_queries = test_queries[: args.iterations]

    # Run Benchmark 1: Query Embeddings
    print(f"  {C_YELLOW}[1/3] Benchmarking Query Embedding Caching...{C_RESET}")
    emb_results = await run_embedding_benchmarks(embeddings_service, test_queries)

    # Run Benchmark 2: Two-Stage Hybrid Retrieval
    print(f"  {C_YELLOW}[2/3] Benchmarking Two-Stage Hybrid Retrieval Caching...{C_RESET}")
    ret_results = await run_retrieval_benchmarks(retrieval_service, test_queries)

    # Run Benchmark 3: Concurrent Multi-Tool Dispatch
    print(f"  {C_YELLOW}[3/3] Benchmarking Concurrent Multi-Tool Dispatch...{C_RESET}")
    tool_results = await run_concurrent_tools_benchmark(tool_registry)

    # Print Formatted Results Table
    print(f"\n{C_BOLD}Benchmark Results Summary:{C_RESET}\n")
    header = (
        f"{'Subsystem / Benchmark':<32} | {'Cold P50':<10} | "
        f"{'Warm P50':<10} | {'Speedup':<9} | {'Hit Ratio':<10}"
    )
    print(header)
    print("-" * len(header))

    emb_row = (
        f"{'Query Embedding Cache':<34} | "
        f"{emb_results['cold']['p50']:<7.2f} ms | "
        f"{emb_results['warm']['p50']:<7.2f} ms | "
        f"{emb_results['speedup_ratio']:<7.1f}x | "
        f"{emb_results['cache_stats'].get('hit_ratio', 0.0) * 100:.1f}%"
    )
    print(emb_row)

    ret_row = (
        f"{'Hybrid Retrieval Cache':<34} | "
        f"{ret_results['cold']['p50']:<7.2f} ms | "
        f"{ret_results['warm']['p50']:<7.2f} ms | "
        f"{ret_results['speedup_ratio']:<7.1f}x | "
        f"{ret_results['cache_stats'].get('hit_ratio', 0.0) * 100:.1f}%"
    )
    print(ret_row)

    tool_row = (
        f"{'Multi-Tool Concurrent Dispatch':<34} | "
        f"{tool_results['sequential_ms']:<7.2f} ms | "
        f"{tool_results['concurrent_ms']:<7.2f} ms | "
        f"{tool_results['speedup_ratio']:<7.1f}x | "
        f"{'N/A':<10}"
    )
    print(tool_row)

    print(f"\n{C_GREEN}[PASS] Performance benchmark completed successfully.{C_RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())

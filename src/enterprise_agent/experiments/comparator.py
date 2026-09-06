"""Run comparison engine calculating metric deltas and Pareto-optimal trade-offs."""

from enterprise_agent.schemas.experiments import (
    ExperimentRun,
    MetricDelta,
    RunComparison,
)


class RunComparisonEngine:
    """Computes comparative scorecards, percentage improvements, and leaders across runs."""

    @staticmethod
    def compare_runs(
        runs: list[ExperimentRun],
        baseline_run_id: str | None = None,
    ) -> RunComparison:
        """Compare candidate runs against a designated baseline run."""
        if len(runs) < 2:
            raise ValueError("Comparison requires at least two experiment runs.")

        # Resolve baseline
        baseline: ExperimentRun
        candidate_runs: list[ExperimentRun] = []

        if baseline_run_id:
            found_baseline = next((r for r in runs if r.run_id == baseline_run_id), None)
            if not found_baseline:
                raise ValueError(f"Baseline run '{baseline_run_id}' not found in provided runs.")
            baseline = found_baseline
            candidate_runs = [r for r in runs if r.run_id != baseline_run_id]
        else:
            baseline = runs[0]
            candidate_runs = runs[1:]

        # Collect all metric keys
        all_metric_keys: set[str] = set(baseline.metrics.keys())
        for cand in candidate_runs:
            all_metric_keys.update(cand.metrics.keys())
        all_metric_keys.add("total_latency_ms")

        metric_deltas: dict[str, dict[str, MetricDelta]] = {}
        for cand in candidate_runs:
            metric_deltas[cand.run_id] = {}
            for metric in sorted(all_metric_keys):
                if metric == "total_latency_ms":
                    b_val = baseline.total_latency_ms
                    c_val = cand.total_latency_ms
                    improved = c_val < b_val
                else:
                    b_val = baseline.metrics.get(metric, 0.0)
                    c_val = cand.metrics.get(metric, 0.0)
                    improved = c_val > b_val

                abs_delta = round(c_val - b_val, 4)
                pct_delta = round((abs_delta / b_val * 100.0), 2) if b_val != 0.0 else 0.0

                metric_deltas[cand.run_id][metric] = MetricDelta(
                    baseline_val=round(b_val, 4),
                    candidate_val=round(c_val, 4),
                    absolute_delta=abs_delta,
                    percentage_delta=pct_delta,
                    improved=improved,
                )

        # Identify leader per metric
        best_runs_per_metric: dict[str, str] = {}
        for metric in sorted(all_metric_keys):
            if metric == "total_latency_ms":
                leader = min(runs, key=lambda r: r.total_latency_ms)
            else:
                leader = max(runs, key=lambda r: r.metrics.get(metric, 0.0))
            best_runs_per_metric[metric] = leader.run_id

        return RunComparison(
            baseline_run=baseline,
            candidate_runs=candidate_runs,
            metric_deltas=metric_deltas,
            best_runs_per_metric=best_runs_per_metric,
        )

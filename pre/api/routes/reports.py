from __future__ import annotations

from typing import Any

from pre.benchmarks.runner import run_benchmark
from pre.demo.runner import build_mode_cards


def reports_route(dataset: str = "nyc_taxi", artifact_root: str = "artifacts") -> dict[str, Any]:
    benchmark = run_benchmark(
        dataset=dataset,
        models=["lstm_gaussian", "lgbm_quantile"],
        horizon=24,
        context_length=168,
        artifact_root=artifact_root,
    )
    return {
        "benchmark": benchmark,
        "modes": build_mode_cards(),
    }

from __future__ import annotations

import argparse
import json

from pre.benchmarks.runner import run_benchmark


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(prog="pre-benchmark")
    parser.add_argument("--dataset", default="nyc_taxi")
    parser.add_argument("--models", default="lstm_gaussian,lgbm_quantile")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=168)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    report = run_benchmark(
        dataset=args.dataset,
        models=models,
        horizon=args.horizon,
        context_length=args.context_length,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    benchmark_main()

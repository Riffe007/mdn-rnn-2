from __future__ import annotations

from pathlib import Path
from typing import Any

from pre.registry.artifacts import ensure_artifact_dir, save_json
from pre.train.trainer import Trainer


def run_benchmark(
    dataset: str,
    models: list[str],
    horizon: int,
    context_length: int,
    artifact_root: str = "artifacts",
) -> dict[str, Any]:
    trainer = Trainer()
    runs: list[dict[str, Any]] = []

    for model in models:
        result = trainer.train(
            dataset=dataset,
            model=model,
            horizon=horizon,
            context_length=context_length,
            artifact_root=artifact_root,
        )
        run_payload = {
            "run_id": result.run_id,
            "dataset": dataset,
            "model": model,
            "artifact_path": result.artifact_path,
            "metrics": result.summary["metrics"],
        }
        runs.append(run_payload)

    leaderboard = sorted(
        runs,
        key=lambda item: (
            item["metrics"]["crps"],
            item["metrics"]["rmse"],
            item["metrics"]["mae"],
        ),
    )

    report = {
        "dataset": dataset,
        "horizon": horizon,
        "context_length": context_length,
        "models": models,
        "leaderboard": leaderboard,
    }

    bench_id = f"benchmark-{dataset}-h{horizon}-c{context_length}"
    bench_dir = ensure_artifact_dir(Path(artifact_root), bench_id)
    save_json(bench_dir / "leaderboard.json", report)
    (bench_dir / "leaderboard.md").write_text(to_markdown(report), encoding="utf-8")
    report["artifact_path"] = str(bench_dir)
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Leaderboard",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Horizon: `{report['horizon']}`",
        f"Context Length: `{report['context_length']}`",
        "",
        "| rank | model | crps | rmse | mae | coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(report["leaderboard"], start=1):
        m = row["metrics"]
        lines.append(
            f"| {rank} | {row['model']} | {m['crps']:.6f} | "
            f"{m['rmse']:.6f} | {m['mae']:.6f} | {m['coverage']:.6f} |"
        )
    return "\n".join(lines)

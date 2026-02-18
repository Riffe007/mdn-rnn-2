from __future__ import annotations

from pathlib import Path

from pre.benchmarks.runner import run_benchmark
from pre.demo.runner import build_mode_cards, run_all_demos, run_demo


def test_benchmark_runner_generates_leaderboard(tmp_path: Path) -> None:
    report = run_benchmark(
        dataset="nyc_taxi",
        models=["lstm_gaussian", "lgbm_quantile"],
        horizon=12,
        context_length=72,
        artifact_root=str(tmp_path),
    )
    assert report["leaderboard"]
    bench_dir = Path(report["artifact_path"])
    assert (bench_dir / "leaderboard.json").exists()
    assert (bench_dir / "leaderboard.md").exists()


def test_demo_mode_outputs_payload_and_artifacts(tmp_path: Path) -> None:
    payload = run_demo(mode="telemetry", artifact_root=str(tmp_path))
    assert payload["mode"] == "telemetry"
    assert "tail_risk_score" in payload
    assert len(payload["visuals"]["horizon"]) == payload["horizon"]
    assert len(payload["visuals"]["bands"]["p50"]) == payload["horizon"]
    assert len(payload["visuals"]["fan"]["p95"]) == payload["horizon"]
    assert payload["visuals"]["calibration_bins"]
    assert payload["visuals"]["tail_risk_heatmap"]
    mode_dir = Path(payload["demo_artifact_path"])
    assert (mode_dir / "demo.json").exists()
    assert (mode_dir / "demo.md").exists()


def test_run_all_demos_and_cards(tmp_path: Path) -> None:
    cards = build_mode_cards()
    assert len(cards) >= 5

    payload = run_all_demos(artifact_root=str(tmp_path))
    assert payload["count"] >= 5

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pre.demo.modes import DEMO_MODES
from pre.registry.artifacts import ensure_artifact_dir, save_json
from pre.train.trainer import Trainer


def _tail_risk_from_metrics(metrics: dict[str, float]) -> float:
    risk = metrics["crps"] + (1.0 - metrics["coverage"])
    return float(max(0.0, min(1.0, risk)))


def _regime_score_from_metrics(metrics: dict[str, float]) -> float:
    score = 0.5 * metrics["rmse"] + 0.5 * metrics["mae"]
    return float(score)


def run_demo(mode: str, artifact_root: str = "artifacts") -> dict[str, Any]:
    if mode not in DEMO_MODES:
        supported = ", ".join(sorted(DEMO_MODES))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")

    spec = DEMO_MODES[mode]
    result = Trainer().train(
        dataset=spec.dataset,
        model=spec.model,
        horizon=spec.horizon,
        context_length=spec.context_length,
        artifact_root=artifact_root,
    )

    metrics = result.summary["metrics"]
    payload = {
        "mode": spec.slug,
        "title": spec.title,
        "description": spec.description,
        "engine": "PRE",
        "dataset": spec.dataset,
        "model": spec.model,
        "horizon": spec.horizon,
        "context_length": spec.context_length,
        "metrics": metrics,
        "tail_risk_score": _tail_risk_from_metrics(metrics),
        "regime_shift_score": _regime_score_from_metrics(metrics),
        "uncertainty_bands": result.summary["visuals"]["bands"],
        "visuals": result.summary["visuals"],
        "artifact_path": result.artifact_path,
    }

    mode_dir = ensure_artifact_dir(Path(artifact_root), f"demo-{spec.slug}")
    save_json(mode_dir / "demo.json", payload)
    (mode_dir / "demo.md").write_text(to_markdown(payload), encoding="utf-8")
    payload["demo_artifact_path"] = str(mode_dir)
    return payload


def run_all_demos(artifact_root: str = "artifacts") -> dict[str, Any]:
    results = [run_demo(mode=mode, artifact_root=artifact_root) for mode in sorted(DEMO_MODES)]
    return {"engine": "PRE", "count": len(results), "modes": results}


def to_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {payload['title']}",
            "",
            payload["description"],
            "",
            f"- Dataset: `{payload['dataset']}`",
            f"- Model: `{payload['model']}`",
            f"- Horizon: `{payload['horizon']}`",
            f"- Tail Risk Score: `{payload['tail_risk_score']:.4f}`",
            f"- Regime Shift Score: `{payload['regime_shift_score']:.4f}`",
            "",
            "## Metrics",
            "",
            f"- MAE: `{payload['metrics']['mae']:.6f}`",
            f"- RMSE: `{payload['metrics']['rmse']:.6f}`",
            f"- NLL: `{payload['metrics']['nll']:.6f}`",
            f"- CRPS: `{payload['metrics']['crps']:.6f}`",
            f"- Coverage: `{payload['metrics']['coverage']:.6f}`",
        ]
    )


def build_mode_cards() -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for mode in sorted(DEMO_MODES):
        spec = DEMO_MODES[mode]
        cards.append(
            {
                "mode": spec.slug,
                "title": spec.title,
                "description": spec.description,
                "dataset": spec.dataset,
                "model": spec.model,
            }
        )
    return cards

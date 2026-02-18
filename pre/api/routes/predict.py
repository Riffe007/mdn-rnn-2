from __future__ import annotations

from typing import Any

from pre.demo.runner import run_demo


def predict_route(mode: str = "demand", artifact_root: str = "artifacts") -> dict[str, Any]:
    """Mode-first inference-style payload for UI demos."""
    return run_demo(mode=mode, artifact_root=artifact_root)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def artifact_path(root: Path, run_id: str) -> Path:
    return root / run_id


def ensure_artifact_dir(root: Path, run_id: str) -> Path:
    path = artifact_path(root, run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecord:
    model_name: str
    version: str
    run_id: str

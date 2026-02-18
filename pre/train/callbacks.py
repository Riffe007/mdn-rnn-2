from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointConfig:
    save_every_n_steps: int = 100
    keep_last_n: int = 3

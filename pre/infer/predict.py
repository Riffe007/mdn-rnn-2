from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PredictiveDistribution:
    """Unified probabilistic output contract for all backends."""

    horizon: np.ndarray
    quantiles: dict[float, np.ndarray]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    samples: np.ndarray | None = None
    tail_risk: dict[str, np.ndarray] = field(default_factory=dict)
    regime_score: np.ndarray | None = None
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def quantile(self, level: float) -> np.ndarray:
        if level not in self.quantiles:
            raise KeyError(f"Missing quantile {level}")
        return self.quantiles[level]

    def interval(self, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        return self.quantile(low), self.quantile(high)

    @classmethod
    def from_samples(
        cls,
        horizon: np.ndarray,
        samples: np.ndarray,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
        metadata: dict[str, str | float | int | bool] | None = None,
    ) -> "PredictiveDistribution":
        if samples.ndim != 2:
            raise ValueError("samples must be shaped [num_samples, horizon]")
        quantiles = {q: np.quantile(samples, q=q, axis=0) for q in quantile_levels}
        return cls(
            horizon=horizon,
            quantiles=quantiles,
            mean=samples.mean(axis=0),
            std=samples.std(axis=0),
            samples=samples,
            metadata=metadata or {},
        )

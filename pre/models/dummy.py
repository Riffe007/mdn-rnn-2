from __future__ import annotations

import numpy as np

from pre.infer.predict import PredictiveDistribution


class DummyModel:
    """Baseline model that repeats the last observed value."""

    name = "dummy"
    horizon: int | None = None
    center_: np.ndarray | None = None
    spread_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _ = x
        self.horizon = y.shape[1]
        self.center_ = y.mean(axis=0)
        spread = y.std(axis=0)
        self.spread_ = np.where(spread < 1e-6, 1.0, spread)

    def predict(self, x: np.ndarray, horizon: int) -> PredictiveDistribution:
        if x.ndim == 1:
            x = x[None, :]

        if self.center_ is None or self.spread_ is None or self.horizon is None:
            last_values = x[:, -1]
            center = np.full((x.shape[0], horizon), float(np.mean(last_values)))
            spread = np.full((x.shape[0], horizon), 1.0)
        else:
            if horizon != self.horizon:
                raise ValueError(
                    f"Requested horizon {horizon} does not match fitted horizon {self.horizon}"
                )
            center = np.repeat(self.center_[None, :], x.shape[0], axis=0)
            spread = np.repeat(self.spread_[None, :], x.shape[0], axis=0)

        return PredictiveDistribution(
            horizon=np.arange(horizon),
            mean=center,
            std=spread,
            quantiles={0.1: center - spread, 0.5: center, 0.9: center + spread},
            metadata={"model": self.name},
        )

    def artifact_state(self) -> dict[str, np.ndarray]:
        if self.center_ is None or self.spread_ is None or self.horizon is None:
            raise ValueError("Model must be fit before serialization")
        return {
            "center": self.center_,
            "spread": self.spread_,
            "horizon": np.array([self.horizon]),
        }

from __future__ import annotations

import numpy as np

from pre.infer.predict import PredictiveDistribution


class TFTQuantileModel:
    name = "tft_quantile"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _ = (x, y)

    def predict(self, x: np.ndarray, horizon: int) -> PredictiveDistribution:
        center = np.full(horizon, float(np.mean(x[:, -1])))
        spread = np.linspace(0.8, 1.2, horizon)
        return PredictiveDistribution(
            horizon=np.arange(horizon),
            mean=center,
            std=spread,
            quantiles={0.1: center - spread, 0.5: center, 0.9: center + spread},
            metadata={"model": self.name},
        )

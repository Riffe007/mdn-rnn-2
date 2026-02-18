from __future__ import annotations

from typing import Protocol

import numpy as np

from pre.infer.predict import PredictiveDistribution


class ForecastModel(Protocol):
    """Model interface used by trainer and inference layers."""

    name: str

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train model."""

    def predict(self, x: np.ndarray, horizon: int) -> PredictiveDistribution:
        """Predict distribution for each step in the forecast horizon."""

from __future__ import annotations

import numpy as np

from pre.infer.predict import PredictiveDistribution


class LSTMGaussianModel:
    """Gaussian forecaster with linear autoregressive fitting (offline-safe placeholder)."""

    name = "lstm_gaussian"

    def __init__(self) -> None:
        self.horizon: int | None = None
        self.coefficients_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.residual_std_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D")

        aug_x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        coef, *_ = np.linalg.lstsq(aug_x, y, rcond=None)
        pred = aug_x @ coef
        residual = y - pred

        self.horizon = y.shape[1]
        self.coefficients_ = coef[:-1, :]
        self.bias_ = coef[-1, :]
        std = residual.std(axis=0)
        self.residual_std_ = np.where(std < 1e-6, 1.0, std)

    def predict(self, x: np.ndarray, horizon: int) -> PredictiveDistribution:
        if x.ndim == 1:
            x = x[None, :]
        if (
            self.coefficients_ is None
            or self.bias_ is None
            or self.residual_std_ is None
            or self.horizon is None
        ):
            raise ValueError("Model is not fitted")
        if horizon != self.horizon:
            raise ValueError(
                f"Requested horizon {horizon} does not match fitted horizon {self.horizon}"
            )

        mean = x @ self.coefficients_ + self.bias_
        std = np.repeat(self.residual_std_[None, :], x.shape[0], axis=0)
        z = 1.28155

        return PredictiveDistribution(
            horizon=np.arange(horizon),
            mean=mean,
            std=std,
            quantiles={
                0.1: mean - z * std,
                0.5: mean,
                0.9: mean + z * std,
            },
            metadata={"model": self.name},
        )

    def artifact_state(self) -> dict[str, np.ndarray]:
        if (
            self.coefficients_ is None
            or self.bias_ is None
            or self.residual_std_ is None
            or self.horizon is None
        ):
            raise ValueError("Model must be fit before serialization")
        return {
            "coefficients": self.coefficients_,
            "bias": self.bias_,
            "residual_std": self.residual_std_,
            "horizon": np.array([self.horizon]),
        }

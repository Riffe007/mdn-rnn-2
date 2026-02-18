from __future__ import annotations

import numpy as np

from pre.infer.predict import PredictiveDistribution


class LGBMQuantileModel:
    """LightGBM-like quantile baseline using linear fit + residual quantiles."""

    name = "lgbm_quantile"

    def __init__(self) -> None:
        self.horizon: int | None = None
        self.coefficients_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.residual_q10_: np.ndarray | None = None
        self.residual_q90_: np.ndarray | None = None

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
        self.residual_q10_ = np.quantile(residual, 0.1, axis=0)
        self.residual_q90_ = np.quantile(residual, 0.9, axis=0)

    def predict(self, x: np.ndarray, horizon: int) -> PredictiveDistribution:
        if x.ndim == 1:
            x = x[None, :]
        if (
            self.coefficients_ is None
            or self.bias_ is None
            or self.residual_q10_ is None
            or self.residual_q90_ is None
            or self.horizon is None
        ):
            raise ValueError("Model is not fitted")
        if horizon != self.horizon:
            raise ValueError(
                f"Requested horizon {horizon} does not match fitted horizon {self.horizon}"
            )

        median = x @ self.coefficients_ + self.bias_
        q10 = median + self.residual_q10_[None, :]
        q90 = median + self.residual_q90_[None, :]
        std = np.maximum((q90 - q10) / (2.0 * 1.28155), 1e-6)

        return PredictiveDistribution(
            horizon=np.arange(horizon),
            mean=median,
            std=std,
            quantiles={0.1: q10, 0.5: median, 0.9: q90},
            metadata={"model": self.name},
        )

    def artifact_state(self) -> dict[str, np.ndarray]:
        if (
            self.coefficients_ is None
            or self.bias_ is None
            or self.residual_q10_ is None
            or self.residual_q90_ is None
            or self.horizon is None
        ):
            raise ValueError("Model must be fit before serialization")
        return {
            "coefficients": self.coefficients_,
            "bias": self.bias_,
            "residual_q10": self.residual_q10_,
            "residual_q90": self.residual_q90_,
            "horizon": np.array([self.horizon]),
        }

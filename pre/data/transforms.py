from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def cyclical_time_features(indices: np.ndarray, period: int) -> np.ndarray:
    """Encode scalar time indices as sin/cos cyclical features."""
    if period <= 0:
        raise ValueError("period must be positive")
    angles = 2.0 * np.pi * (indices % period) / period
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1)

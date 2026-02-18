from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    expected: float
    observed: float
    count: int


def quantile_reliability_bins(
    y_true: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
) -> list[ReliabilityBin]:
    bins: list[ReliabilityBin] = []
    for q in sorted(quantile_predictions):
        pred = quantile_predictions[q]
        observed = float(np.mean(y_true <= pred))
        bins.append(ReliabilityBin(expected=q, observed=observed, count=int(y_true.size)))
    return bins


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

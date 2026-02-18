from __future__ import annotations

import math

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> float:
    safe_std = np.clip(std, eps, None)
    z = (y_true - mean) / safe_std
    ll = -0.5 * np.log(2.0 * np.pi) - np.log(safe_std) - 0.5 * (z**2)
    return float(-np.mean(ll))


def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (z**2)) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_values = np.vectorize(math.erf)(z / math.sqrt(2.0))
    return 0.5 * (1.0 + erf_values)


def crps_gaussian(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    safe_std = np.clip(std, eps, None)
    z = (y_true - mean) / safe_std
    cdf = _standard_normal_cdf(z)
    pdf = _standard_normal_pdf(z)
    crps = safe_std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

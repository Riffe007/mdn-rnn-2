from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class WindowedDataset:
    features: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class DatasetSplit:
    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset


@dataclass(frozen=True)
class WindowSpec:
    context_length: int
    horizon: int
    stride: int = 1


class BacktestFold(NamedTuple):
    train_slice: slice
    eval_slice: slice


class DatasetAdapter(Protocol):
    """Contract for converting a source dataset into model-ready windows."""

    def load(self) -> np.ndarray:
        """Load raw data as a 2D numeric array."""

    def make_windows(self, horizon: int, context_length: int) -> WindowedDataset:
        """Create supervised context/horizon windows."""


def make_supervised_windows(values: np.ndarray, spec: WindowSpec) -> WindowedDataset:
    """Convert a 1D or single-column series into context/horizon supervised windows."""
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError("values must be 1D or a single-column 2D array")
        series = values.squeeze(1)
    elif values.ndim == 1:
        series = values
    else:
        raise ValueError("values must be 1D or 2D")

    window_count = len(series) - spec.context_length - spec.horizon + 1
    if window_count <= 0:
        raise ValueError("Insufficient samples for requested context_length + horizon")

    x = np.zeros((window_count, spec.context_length), dtype=float)
    y = np.zeros((window_count, spec.horizon), dtype=float)
    t = np.zeros(window_count, dtype=int)

    out_idx = 0
    for start in range(0, window_count, spec.stride):
        x[out_idx] = series[start : start + spec.context_length]
        y[out_idx] = series[
            start + spec.context_length : start + spec.context_length + spec.horizon
        ]
        t[out_idx] = start + spec.context_length
        out_idx += 1

    return WindowedDataset(features=x[:out_idx], targets=y[:out_idx], timestamps=t[:out_idx])


def temporal_train_val_test_split(
    dataset: WindowedDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetSplit:
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    n = dataset.features.shape[0]
    test_n = max(1, int(n * test_ratio))
    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n - test_n
    if train_n < 1:
        raise ValueError("Not enough windows for train/val/test split")

    def _slice(start: int, end: int) -> WindowedDataset:
        return WindowedDataset(
            features=dataset.features[start:end],
            targets=dataset.targets[start:end],
            timestamps=dataset.timestamps[start:end],
        )

    train = _slice(0, train_n)
    val = _slice(train_n, train_n + val_n)
    test = _slice(train_n + val_n, n)
    return DatasetSplit(train=train, val=val, test=test)


def rolling_backtest_folds(
    num_windows: int,
    initial_train_size: int,
    eval_size: int,
    step: int,
) -> list[BacktestFold]:
    if initial_train_size < 1 or eval_size < 1 or step < 1:
        raise ValueError("initial_train_size, eval_size, and step must be >= 1")
    if initial_train_size + eval_size > num_windows:
        return []

    folds: list[BacktestFold] = []
    train_end = initial_train_size
    while train_end + eval_size <= num_windows:
        folds.append(
            BacktestFold(
                train_slice=slice(0, train_end),
                eval_slice=slice(train_end, train_end + eval_size),
            )
        )
        train_end += step
    return folds

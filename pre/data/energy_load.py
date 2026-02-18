from __future__ import annotations

import numpy as np

from pre.data.base import (
    DatasetAdapter,
    DatasetSplit,
    WindowSpec,
    WindowedDataset,
    make_supervised_windows,
    temporal_train_val_test_split,
)


class EnergyLoadAdapter(DatasetAdapter):
    """Synthetic energy load with daily and weekly seasonality."""

    def __init__(self, periods: int = 24 * 60, seed: int = 17) -> None:
        self.periods = periods
        self.seed = seed

    def load(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.periods)
        daily = 30 * np.sin(2 * np.pi * t / 24)
        weekly = 12 * np.sin(2 * np.pi * t / (24 * 7))
        weather = 7 * np.sin(2 * np.pi * t / (24 * 20))
        noise = rng.normal(0.0, 1.8, size=self.periods)
        signal = 220 + daily + weekly + weather + noise
        return signal.reshape(-1, 1)

    def make_windows(self, horizon: int, context_length: int, stride: int = 1) -> WindowedDataset:
        spec = WindowSpec(context_length=context_length, horizon=horizon, stride=stride)
        return make_supervised_windows(values=self.load(), spec=spec)

    def split(
        self,
        horizon: int,
        context_length: int,
        stride: int = 1,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplit:
        windows = self.make_windows(horizon=horizon, context_length=context_length, stride=stride)
        return temporal_train_val_test_split(windows, val_ratio=val_ratio, test_ratio=test_ratio)

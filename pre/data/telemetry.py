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


class TelemetryAdapter(DatasetAdapter):
    """Synthetic telemetry stream with spikes and occasional regime changes."""

    def __init__(self, points: int = 24 * 45, seed: int = 11) -> None:
        self.points = points
        self.seed = seed

    def load(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.points)
        base = 60 + 8 * np.sin(2 * np.pi * t / 24)
        trend = 0.004 * t
        noise = rng.normal(0.0, 1.2, size=self.points)

        spikes = np.zeros(self.points)
        spike_idx = rng.choice(self.points, size=max(1, self.points // 40), replace=False)
        spikes[spike_idx] = rng.uniform(8, 18, size=spike_idx.size)

        regime = np.where(t > int(self.points * 0.65), 6.0, 0.0)
        signal = base + trend + noise + spikes + regime
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

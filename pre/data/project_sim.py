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


class ProjectTimelineAdapter(DatasetAdapter):
    """Synthetic project-duration stream with risk events and resource contention."""

    def __init__(self, n_tasks: int = 1800, seed: int = 23) -> None:
        self.n_tasks = n_tasks
        self.seed = seed

    def load(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        baseline = rng.lognormal(mean=2.0, sigma=0.35, size=self.n_tasks)
        risk_events = rng.binomial(n=1, p=0.2, size=self.n_tasks)
        event_impact = rng.uniform(0.0, 4.5, size=self.n_tasks)
        contention = 0.8 * np.sin(2 * np.pi * np.arange(self.n_tasks) / 45)
        durations = baseline + risk_events * event_impact + contention
        return durations.reshape(-1, 1)

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

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


class NYCTaxiAdapter(DatasetAdapter):
    """Synthetic placeholder until dataset ingestion is implemented."""

    def load(self) -> np.ndarray:
        t = np.arange(24 * 60)
        daily = 20 * np.sin(2 * np.pi * t / 24)
        weekly = 8 * np.sin(2 * np.pi * t / (24 * 7))
        trend = 0.01 * t
        signal = 100 + daily + weekly + trend
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

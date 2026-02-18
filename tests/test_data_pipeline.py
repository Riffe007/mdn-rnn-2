from __future__ import annotations

from pathlib import Path

import numpy as np

from pre.data.base import (
    WindowSpec,
    make_supervised_windows,
    rolling_backtest_folds,
    temporal_train_val_test_split,
)
from pre.data.nyc_taxi import NYCTaxiAdapter
from pre.train.trainer import Trainer


def test_windowing_supports_multiple_horizons() -> None:
    adapter = NYCTaxiAdapter()
    for horizon in (6, 12, 24):
        windows = adapter.make_windows(horizon=horizon, context_length=48)
        assert windows.features.shape[1] == 48
        assert windows.targets.shape[1] == horizon
        assert windows.features.shape[0] == windows.targets.shape[0]


def test_temporal_split_preserves_order_and_sizes() -> None:
    series = np.arange(200, dtype=float)
    windows = make_supervised_windows(series, WindowSpec(context_length=24, horizon=6))
    split = temporal_train_val_test_split(windows, val_ratio=0.1, test_ratio=0.1)

    total = (
        split.train.features.shape[0]
        + split.val.features.shape[0]
        + split.test.features.shape[0]
    )
    assert total == windows.features.shape[0]
    assert split.train.timestamps.max() < split.val.timestamps.min()
    assert split.val.timestamps.max() < split.test.timestamps.min()


def test_rolling_backtest_folds_generated() -> None:
    folds = rolling_backtest_folds(
        num_windows=120,
        initial_train_size=60,
        eval_size=12,
        step=6,
    )
    assert len(folds) > 0
    assert folds[0].train_slice == slice(0, 60)
    assert folds[0].eval_slice == slice(60, 72)


def test_trainer_runs_shape_checks_with_dummy_model() -> None:
    result = Trainer().train(
        dataset="nyc_taxi",
        model="dummy",
        horizon=12,
        context_length=72,
    )
    assert result.summary["shape_checks_passed"] is True
    assert result.summary["batch_shapes"]["train"]["targets"][1] == 12


def test_lstm_gaussian_training_writes_artifacts_and_metrics(  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    result = Trainer().train(
        dataset="nyc_taxi",
        model="lstm_gaussian",
        horizon=12,
        context_length=72,
        artifact_root=str(tmp_path),
    )
    artifact_dir = Path(result.artifact_path)
    assert (artifact_dir / "config.json").exists()
    assert (artifact_dir / "scaler.npz").exists()
    assert (artifact_dir / "model.npz").exists()
    assert (artifact_dir / "report.json").exists()
    assert (artifact_dir / "report.md").exists()

    for key in ("mae", "rmse", "nll", "crps", "coverage"):
        assert key in result.summary["metrics"]


def test_lgbm_quantile_runs_on_telemetry_dataset(  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    result = Trainer().train(
        dataset="telemetry",
        model="lgbm_quantile",
        horizon=24,
        context_length=120,
        artifact_root=str(tmp_path),
    )
    assert result.summary["shape_checks_passed"] is True
    assert result.summary["window_counts"]["train"] > 0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pre.data.base import DatasetAdapter, rolling_backtest_folds
from pre.data.energy_load import EnergyLoadAdapter
from pre.data.nyc_taxi import NYCTaxiAdapter
from pre.data.project_sim import ProjectTimelineAdapter
from pre.data.telemetry import TelemetryAdapter
from pre.data.transforms import StandardScaler
from pre.eval.reports import build_report, to_markdown
from pre.models.dummy import DummyModel
from pre.models.lgbm_quantile import LGBMQuantileModel
from pre.models.lstm_gaussian import LSTMGaussianModel
from pre.registry.artifacts import ensure_artifact_dir, save_json, save_npz


def _assert_shape_consistency(
    features_shape: tuple[int, ...],
    targets_shape: tuple[int, ...],
    horizon: int,
) -> None:
    if len(features_shape) != 2:
        raise ValueError(f"features must be rank 2, got shape {features_shape}")
    if len(targets_shape) != 2:
        raise ValueError(f"targets must be rank 2, got shape {targets_shape}")
    if features_shape[0] != targets_shape[0]:
        raise ValueError(f"batch count mismatch: {features_shape} vs {targets_shape}")
    if targets_shape[1] != horizon:
        raise ValueError(f"target horizon mismatch: expected {horizon}, got {targets_shape[1]}")


def _resolve_dataset(dataset: str) -> DatasetAdapter:
    adapters: dict[str, DatasetAdapter] = {
        "nyc_taxi": NYCTaxiAdapter(),
        "telemetry": TelemetryAdapter(),
        "energy_load": EnergyLoadAdapter(),
        "project_sim": ProjectTimelineAdapter(),
    }
    if dataset not in adapters:
        supported = ", ".join(sorted(adapters))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")
    return adapters[dataset]


def _resolve_model(model: str) -> DummyModel | LSTMGaussianModel | LGBMQuantileModel:
    if model == "dummy":
        return DummyModel()
    if model == "lstm_gaussian":
        return LSTMGaussianModel()
    if model == "lgbm_quantile":
        return LGBMQuantileModel()
    raise ValueError("Unsupported model. Supported models: dummy, lstm_gaussian, lgbm_quantile")


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_name: str
    artifact_path: str
    summary: dict[str, Any]


class Trainer:
    """Common train entrypoint used by CLI and API layers."""

    def train(
        self,
        dataset: str,
        model: str,
        horizon: int = 24,
        context_length: int = 168,
        stride: int = 1,
        artifact_root: str = "artifacts",
    ) -> TrainResult:
        adapter = _resolve_dataset(dataset)
        model_impl = _resolve_model(model)

        split = adapter.split(
            horizon=horizon,
            context_length=context_length,
            stride=stride,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        _assert_shape_consistency(split.train.features.shape, split.train.targets.shape, horizon)
        _assert_shape_consistency(split.val.features.shape, split.val.targets.shape, horizon)
        _assert_shape_consistency(split.test.features.shape, split.test.targets.shape, horizon)

        scaler = StandardScaler().fit(split.train.features)
        x_train = scaler.transform(split.train.features)
        x_test = scaler.transform(split.test.features)

        model_impl.fit(x_train, split.train.targets)
        test_dist = model_impl.predict(x_test, horizon=horizon)

        if test_dist.mean is None or test_dist.std is None:
            raise ValueError(
                "Predictive distribution must include mean and std "
                "for report generation"
            )

        report = build_report(
            y_true=split.test.targets.reshape(-1),
            y_pred_mean=test_dist.mean.reshape(-1),
            y_pred_std=test_dist.std.reshape(-1),
            quantiles={q: values.reshape(-1) for q, values in test_dist.quantiles.items()},
        )

        folds = rolling_backtest_folds(
            num_windows=split.train.features.shape[0],
            initial_train_size=max(1, int(split.train.features.shape[0] * 0.6)),
            eval_size=max(1, horizon),
            step=max(1, horizon),
        )[:25]

        run_id = f"{dataset}-{model}-h{horizon}-c{context_length}"
        artifact_dir = ensure_artifact_dir(Path(artifact_root), run_id)

        save_json(
            artifact_dir / "config.json",
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "context_length": context_length,
                "stride": stride,
            },
        )
        if scaler.mean_ is None or scaler.std_ is None:
            raise ValueError("Scaler is not fitted")
        save_npz(
            artifact_dir / "scaler.npz",
            {
                "mean": scaler.mean_,
                "std": scaler.std_,
            },
        )
        save_npz(artifact_dir / "model.npz", model_impl.artifact_state())
        save_json(artifact_dir / "report.json", report)
        (artifact_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")

        summary = {
            "dataset": dataset,
            "model": model,
            "horizon": horizon,
            "context_length": context_length,
            "stride": stride,
            "window_counts": {
                "train": int(split.train.features.shape[0]),
                "val": int(split.val.features.shape[0]),
                "test": int(split.test.features.shape[0]),
            },
            "batch_shapes": {
                "train": {
                    "features": list(split.train.features.shape),
                    "targets": list(split.train.targets.shape),
                },
                "val": {
                    "features": list(split.val.features.shape),
                    "targets": list(split.val.targets.shape),
                },
                "test": {
                    "features": list(split.test.features.shape),
                    "targets": list(split.test.targets.shape),
                },
            },
            "backtest_folds": [
                {
                    "train": [fold.train_slice.start, fold.train_slice.stop],
                    "eval": [fold.eval_slice.start, fold.eval_slice.stop],
                }
                for fold in folds
            ],
            "shape_checks_passed": True,
            "metrics": {
                "mae": float(report["mae"]),
                "rmse": float(report["rmse"]),
                "nll": float(report["nll"]),
                "crps": float(report["crps"]),
                "coverage": float(report["coverage"]),
            },
            "artifacts": ["config.json", "scaler.npz", "model.npz", "report.json", "report.md"],
        }
        return TrainResult(
            run_id=run_id,
            model_name=model,
            artifact_path=str(artifact_dir),
            summary=summary,
        )

    @staticmethod
    def to_dict(result: TrainResult) -> dict[str, Any]:
        return asdict(result)

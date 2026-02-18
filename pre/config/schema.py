from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset: str = Field(default="nyc_taxi")
    horizon: int = Field(default=24, gt=0)
    context_length: int = Field(default=168, gt=0)
    target_column: str = Field(default="target")
    time_column: str = Field(default="timestamp")


class TrainConfig(BaseModel):
    seed: int = 7
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    artifact_dir: Path = Field(default=Path("artifacts"))


class EvalConfig(BaseModel):
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_bins: int = Field(default=10, gt=1)


class ModelConfig(BaseModel):
    name: Literal["dummy", "lstm_gaussian", "lgbm_quantile", "tft_quantile", "mdn_rnn"] = "dummy"
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

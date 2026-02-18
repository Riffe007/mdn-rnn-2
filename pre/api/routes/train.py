from __future__ import annotations

from typing import Any

from pre.train.trainer import Trainer


def train_route(
    dataset: str = "nyc_taxi",
    model: str = "lstm_gaussian",
    horizon: int = 24,
    context_length: int = 168,
    artifact_root: str = "artifacts",
) -> dict[str, Any]:
    result = Trainer().train(
        dataset=dataset,
        model=model,
        horizon=horizon,
        context_length=context_length,
        artifact_root=artifact_root,
    )
    return Trainer.to_dict(result)

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictRequest(BaseModel):
    run_id: str = "latest"


class PredictResponse(BaseModel):
    horizon: list[int]
    quantiles: dict[str, list[float]]

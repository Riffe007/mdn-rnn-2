from __future__ import annotations

from fastapi import FastAPI

from pre.api.routes.health import health
from pre.api.routes.predict import predict_route
from pre.api.routes.reports import reports_route
from pre.api.routes.train import train_route


def create_app() -> FastAPI:
    app = FastAPI(title="Riffe Labs PRE API", version="0.1.0")

    @app.get('/health')
    def health_endpoint() -> dict[str, str]:
        return health()

    @app.post('/train')
    def train_endpoint(
        dataset: str = 'nyc_taxi',
        model: str = 'lstm_gaussian',
        horizon: int = 24,
        context_length: int = 168,
        artifact_root: str = 'artifacts',
    ) -> dict[str, object]:
        return train_route(
            dataset=dataset,
            model=model,
            horizon=horizon,
            context_length=context_length,
            artifact_root=artifact_root,
        )

    @app.get('/predict')
    def predict_endpoint(mode: str = 'demand', artifact_root: str = 'artifacts') -> dict[str, object]:
        return predict_route(mode=mode, artifact_root=artifact_root)

    @app.get('/reports')
    def reports_endpoint(dataset: str = 'nyc_taxi', artifact_root: str = 'artifacts') -> dict[str, object]:
        return reports_route(dataset=dataset, artifact_root=artifact_root)

    return app


app = create_app()

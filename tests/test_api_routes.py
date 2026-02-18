from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pre.api.app import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'


def test_train_predict_reports_routes(tmp_path: Path) -> None:
    client = TestClient(create_app())

    train_response = client.post(
        '/train',
        params={
            'dataset': 'nyc_taxi',
            'model': 'lstm_gaussian',
            'horizon': 12,
            'context_length': 72,
            'artifact_root': str(tmp_path),
        },
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload['summary']['shape_checks_passed'] is True

    predict_response = client.get('/predict', params={'mode': 'demand', 'artifact_root': str(tmp_path)})
    assert predict_response.status_code == 200
    predict_payload = predict_response.json()
    assert predict_payload['mode'] == 'demand'

    reports_response = client.get('/reports', params={'dataset': 'nyc_taxi', 'artifact_root': str(tmp_path)})
    assert reports_response.status_code == 200
    reports_payload = reports_response.json()
    assert 'benchmark' in reports_payload
    assert reports_payload['benchmark']['leaderboard']

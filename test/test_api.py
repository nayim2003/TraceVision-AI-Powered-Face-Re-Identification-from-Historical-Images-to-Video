
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app


def test_root_endpoint():
    with patch(
        "app.api.main.build_tracevision_pipeline"
    ):

        with TestClient(app) as client:

            response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "TraceVision"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint():
    with patch(
        "app.api.main.build_tracevision_pipeline"
    ):

        with TestClient(app) as client:

            response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "TraceVision"
    assert data["version"] == "0.1.0"


def test_tracks_endpoint_returns_list():
    with patch(
        "app.api.main.build_tracevision_pipeline"
    ):

        with TestClient(app) as client:

            response = client.get(
                "/results/tracks"
            )

    assert response.status_code == 200
    assert isinstance(response.json(), list)

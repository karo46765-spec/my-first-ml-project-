import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Ficture initializing the fastapi testclient withincontext"""
    with TestClient(app) as client:
        yield client


def test_read_root_endpoint(client):
    """verifies that the root endpoint"""
    responce = client.get("/")
    assert responce.status_code == 200
    assert responce.json()["status"] == "online"


def test_predict_endpoint_valid_applicant(client):
    """verifies that post /predict process"""
    payload = {"credit_score": 750, "dti_ratio": 0.15}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["Approved", "Denied"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["raw_prediction"] in [0, 1]


def test_predict_endpoint_invalid_applicant_bounds(client):
    """verifies that pydantic rejects"""
    invalid_payload = {"credit_score": 150, "dti_ratio": 0.15}
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_health_check_endpoint(client):
    """Verifies liveness check probe."""
    responce = client.get("/health")
    assert responce.status_code == 200
    assert responce.json()["status"] == "alive"


def test_readiness_check_endpoint(client):
    """verifies readiness check probe."""
    responce = client.get("/ready")
    assert responce.status_code == 200
    assert responce.json()["status"] == "ready"
    assert responce.json()["model_loaded"] is True


def test_process_time_header(client):
    """Verifies custom latency process time middleware is attached"""
    responce = client.get("/health")
    assert responce.status_code == 200
    assert "x-process-time" in responce.headers

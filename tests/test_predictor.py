import pytest
import numpy as nm
from sklearn.linear_model import LogisticRegression
from src.models.predictor_wrapper import creditpredictor


@pytest.fixture
def trained_predictor():
    """Fixture providing instance"""
    x_train = nm.array([[1.5, -1.2], [-1.2, 1.5], [0.0, -0.2]])
    y_train = nm.array([0, 1, 0])

    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)
    return creditpredictor(model)


def test_predictor_schema_output(trained_predictor):
    """Verifies output payload keys and data types."""
    mock_applicant = nm.array([[1.0, -1.0]])
    payload = trained_predictor.predict_structured(mock_applicant)

    assert isinstance(payload, dict)
    assert "confidence" in payload
    assert "status" in payload
    assert "raw_prediction" in payload
    assert 0.0 <= payload["confidence"] <= 1.0


def test_prediction_high_risk_rejection(trained_predictor):
    """Verifies high-risk applicant receives a 'denied' stattus."""
    high_risk_applicant = nm.array([[-3.0, 3.0]])
    payload = trained_predictor.predict_structured(high_risk_applicant)

    assert payload["status"] == "Denied"
    assert payload["raw_prediction"] == 1

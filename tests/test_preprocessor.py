import pytest
import numpy as nm
from src.preprocessing.data_preprocessor import creditprocessor
from src.exception import DataValidationError


@pytest.fixture
def sample_training_data():
    """Fixture providing standard mock training feature"""
    return nm.array([[700.0, 0.2], [550.0, 0.6], [600.0, 0.4]])


@pytest.fixture
def fitted_preprocessor(sample_training_data):
    """Fixture returning already fitted sample data"""
    preprocessor = creditprocessor()
    preprocessor.fit_pipeline(sample_training_data)
    return preprocessor


def test_preprocessor_scaling_transform(fitted_preprocessor, sample_training_data):
    """Verifies data scaling output properties(mean near 0)."""
    scaled_data = fitted_preprocessor.transform_data(sample_training_data)
    assert scaled_data.shape == sample_training_data.shape
    nm.testing.assert_almost_equal(scaled_data.mean(axis=0), [0.0, 0.0], decimal=5)


def test_invalid_input_raises_exception(fitted_preprocessor):
    """verifies 1d array"""
    invalid_input = nm.array([500.0, 0.5])

    with pytest.raises(DataValidationError):
        fitted_preprocessor.transform_data(invalid_input)

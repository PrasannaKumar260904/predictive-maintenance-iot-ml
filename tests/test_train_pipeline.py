"""Unit test for master model training pipeline."""

from src.models.train import get_base_models


def test_get_base_models():
    models = get_base_models()
    assert isinstance(models, dict)
    assert "RandomForest" in models
    assert "LinearRegression" in models

"""Unit tests for ML models, Neural Nets, Registry, and Inference Engine."""


import numpy as np
import pandas as pd

from src.models.neural_net import PyTorchNeuralNetRegressor
from src.models.predict import PredictiveMaintenanceInferenceEngine
from src.models.registry import ModelRegistry


def test_pytorch_neural_net():
    X = np.random.randn(50, 10)
    y = np.random.randn(50)

    model = PyTorchNeuralNetRegressor(input_dim=10, epochs=3, batch_size=16)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (50,)
    assert not np.isnan(preds).any()


def test_model_registry(tmp_path):
    registry = ModelRegistry(models_dir=str(tmp_path))

    mock_model = {"name": "test_estimator"}
    feature_names = ["feat1", "feat2"]
    metrics = {"RMSE": 10.5, "R2": 0.85}

    saved_path = registry.save_model(
        model=mock_model,
        model_name="unit_test_model",
        feature_names=feature_names,
        metrics=metrics,
    )
    assert saved_path.exists()

    loaded_model, loaded_features, _, loaded_metrics = registry.load_model("unit_test_model")
    assert loaded_model["name"] == "test_estimator"
    assert loaded_features == feature_names
    assert loaded_metrics["RMSE"] == 10.5


def test_inference_engine():
    engine = PredictiveMaintenanceInferenceEngine(model_name="best_model")
    df_sample = pd.DataFrame(
        [
            {
                "engine_id": 1,
                "cycle": 100,
                "temperature": 85.0,
                "pressure": 135.0,
                "vibration": 2.1,
                "voltage": 395.0,
                "current": 20.0,
                "humidity": 45.0,
                "power_consumption": 11.5,
                "rpm": 2850.0,
                "torque": 140.0,
                "operating_hours": 2400,
                "error_code": 0,
                "days_since_maintenance": 45,
            }
        ]
    )

    ruls = engine.predict_rul(df_sample)
    assert len(ruls) == 1
    assert ruls[0] >= 0.0

    risk_assess = engine.predict_failure_risk(df_sample)
    assert len(risk_assess) == 1
    assert "predicted_rul_cycles" in risk_assess[0]
    assert "risk_level" in risk_assess[0]

"""Unit tests for deployment pipeline."""

import pandas as pd

from src.deployment.pipeline import DeploymentPipeline


def test_deployment_pipeline():
    pipeline = DeploymentPipeline(model_name="best_model")

    df_sample = pd.DataFrame(
        [
            {
                "engine_id": 1,
                "cycle": 100,
                "temperature": 82.5,
                "pressure": 138.2,
                "vibration": 1.85,
                "voltage": 395.0,
                "current": 19.4,
                "humidity": 46.2,
                "power_consumption": 11.2,
                "rpm": 2850.0,
                "torque": 142.5,
                "operating_hours": 2400,
                "error_code": 0,
                "days_since_maintenance": 45,
            }
        ]
    )

    results = pipeline.run_inference(df_sample)
    assert isinstance(results, list)
    assert len(results) == 1
    assert "predicted_rul_cycles" in results[0]

    explanation = pipeline.explain_prediction(df_sample)
    assert "top_drivers" in explanation

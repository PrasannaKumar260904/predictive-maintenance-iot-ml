"""Production FastAPI REST API Service for Industrial IoT Predictive Maintenance."""

import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchTelemetryInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionOutput,
    SHAPExplanationOutput,
    TelemetryInput,
)
from src.deployment.pipeline import DeploymentPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Predictive Maintenance IoT ML API",
    description="Production REST API for Industrial Equipment Failure Prediction, Remaining Useful Life (RUL) estimation, and SHAP Explainability.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit UI & Enterprise Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Deployment Pipeline Instance
pipeline: DeploymentPipeline = None


@app.on_event("startup")
def startup_event():
    """Initializes ML inference pipeline on server startup."""
    global pipeline
    try:
        pipeline = DeploymentPipeline(model_name="best_model")
        logger.info("FastAPI Application & ML Pipeline initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not load pre-trained model on startup: {e}. Will attempt train/fallback.")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Returns API health status."""
    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["MLOps"])
def get_model_info() -> ModelInfoResponse:
    """Returns active model metadata, feature list, and evaluation metrics."""
    if pipeline is None or pipeline.engine.model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")

    return ModelInfoResponse(
        model_name=type(pipeline.engine.model).__name__,
        num_features=len(pipeline.engine.feature_names),
        feature_names=pipeline.engine.feature_names,
        performance_metrics=pipeline.engine.metrics or {},
    )


@app.post("/predict", response_model=list[PredictionOutput], tags=["Inference"])
def predict(input_data: BatchTelemetryInput) -> list[PredictionOutput]:
    """Predicts Remaining Useful Life (RUL), Failure Risk Level, and Actionable Recommendations."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline unavailable.")

    try:
        records_dict = [r.dict() for r in input_data.records]
        df_input = pd.DataFrame(records_dict)

        results = pipeline.run_inference(df_input)
        response = []

        for item, rec in zip(results, records_dict):
            response.append(
                PredictionOutput(
                    engine_id=rec["engine_id"],
                    cycle=rec["cycle"],
                    predicted_rul_cycles=item["predicted_rul_cycles"],
                    failure_probability=item["failure_probability"],
                    risk_level=item["risk_level"],
                    status_color=item["status_color"],
                    recommended_action=item["recommended_action"],
                )
            )

        return response
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_rul", tags=["Inference"])
def predict_rul(input_data: TelemetryInput) -> dict[str, Any]:
    """Single-record Remaining Useful Life (RUL) regression endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline unavailable.")

    df_input = pd.DataFrame([input_data.dict()])
    ruls = pipeline.engine.predict_rul(df_input)
    predicted_rul = float(ruls[0])

    return {
        "engine_id": input_data.engine_id,
        "cycle": input_data.cycle,
        "predicted_rul_cycles": round(predicted_rul, 2),
    }


@app.post("/explain", response_model=SHAPExplanationOutput, tags=["Explainability"])
def explain_prediction(input_data: TelemetryInput) -> SHAPExplanationOutput:
    """Generates SHAP feature contribution breakdown for a given telemetry snapshot."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline unavailable.")

    try:
        df_input = pd.DataFrame([input_data.dict()])
        explanation = pipeline.explain_prediction(df_input)
        ruls = pipeline.engine.predict_rul(df_input)

        return SHAPExplanationOutput(
            engine_id=input_data.engine_id,
            predicted_rul_cycles=round(float(ruls[0]), 2),
            base_value=explanation["base_value"],
            top_drivers=explanation["top_drivers"],
        )
    except Exception as e:
        logger.error(f"Explainability Error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

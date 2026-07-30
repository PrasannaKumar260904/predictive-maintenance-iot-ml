"""Pydantic data schemas for request and response validation in FastAPI."""

from typing import Any

from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """Single-cycle sensor telemetry input schema."""

    engine_id: int = Field(default=1, description="Engine or machine unit identifier")
    cycle: int = Field(default=100, description="Current operational time cycle")
    temperature: float = Field(default=82.5, description="Temperature reading in °C")
    pressure: float = Field(default=138.2, description="Pressure reading in PSI")
    vibration: float = Field(default=1.85, description="Vibration level in mm/s RMS")
    voltage: float = Field(default=395.0, description="Electrical voltage in V")
    current: float = Field(default=19.4, description="Electrical current in A")
    humidity: float = Field(default=46.2, description="Ambient humidity percentage")
    power_consumption: float = Field(default=11.2, description="Power consumption in kW")
    rpm: float = Field(default=2850.0, description="Rotational speed in RPM")
    torque: float = Field(default=142.5, description="Torque in Nm")
    operating_hours: int = Field(default=2400, description="Cumulative machine operating hours")
    error_code: int = Field(
        default=0, description="Recent diagnostic error code (0: None, 1: Warning, 2+: Fault)"
    )
    days_since_maintenance: int = Field(
        default=45, description="Days elapsed since last maintenance service"
    )


class BatchTelemetryInput(BaseModel):
    """Batch list of sensor telemetry inputs."""

    records: list[TelemetryInput]


class PredictionOutput(BaseModel):
    """Inference response output schema."""

    engine_id: int
    cycle: int
    predicted_rul_cycles: float
    failure_probability: float
    risk_level: str
    status_color: str
    recommended_action: str


class HealthResponse(BaseModel):
    """System health response schema."""

    status: str
    version: str
    environment: str


class ModelInfoResponse(BaseModel):
    """Active model metadata response schema."""

    model_name: str
    num_features: int
    feature_names: list[str]
    performance_metrics: dict[str, float]


class SHAPExplanationOutput(BaseModel):
    """SHAP explanation breakdown response schema."""

    engine_id: int
    predicted_rul_cycles: float
    base_value: float
    top_drivers: list[dict[str, Any]]

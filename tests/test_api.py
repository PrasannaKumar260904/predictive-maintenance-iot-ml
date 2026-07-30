"""Integration tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "version" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    # Should be 200 if initialized or 503 if not
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "feature_names" in data


def test_predict_endpoint():
    payload = {
        "records": [
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
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert "predicted_rul_cycles" in data[0]
        assert "risk_level" in data[0]


def test_predict_rul_endpoint():
    payload = {
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
    response = client.post("/predict_rul", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_rul_cycles" in data


def test_explain_endpoint():
    payload = {
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
    response = client.post("/explain", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "top_drivers" in data

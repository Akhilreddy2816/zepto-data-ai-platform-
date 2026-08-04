"""
pytest Unit Tests for FastAPI Backend REST API Endpoints
"""

from fastapi.testclient import TestClient
import pytest
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "modules" in data


def test_pipeline_run_endpoint():
    response = client.post("/api/v1/pipeline/run", json={"num_items": 15})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "summary" in data


def test_predict_endpoint():
    payload = {
        "customer_tenure_months": 12,
        "order_distance_km": 4.5,
        "item_count": 5,
        "order_value_inr": 500.0,
        "traffic_density": "Medium",
        "weather_condition": "Clear",
        "driver_experience_years": 3,
        "delivery_time_mins": 15.0,
    }
    response = client.post("/api/v1/analytics/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "prediction" in data


def test_rag_chat_endpoint():
    payload = {"query": "How many leaves do employees get?"}
    response = client.post("/api/v1/rag/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "response" in data

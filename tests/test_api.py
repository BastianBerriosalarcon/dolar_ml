"""
Tests para API REST
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test del endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test del health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "features_count" in data


def test_get_features():
    """Test del endpoint de features"""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "total_count" in data


def test_get_metrics():
    """Test del endpoint de métricas"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Puede no tener métricas si el modelo no está entrenado, pero debe responder


def test_predict_endpoint_structure():
    """Test de estructura del endpoint de predicción"""
    # Este test asume que el modelo no está cargado
    # En producción, necesitarías un modelo mock
    sample_features = {
        "features": {
            "Valor_lag_1": 920.5,
            "Valor_lag_7": 918.2,
            "Valor_lag_30": 915.0,
            "MA_7": 919.0,
            "MA_30": 917.5,
            "MA_90": 910.0,
            "day_of_week": 1,
            "month": 3,
            "quarter": 1,
            "year": 2024,
            "ROC_7": 0.25,
            "momentum_7": 2.3
        }
    }
    
    response = client.post("/predict", json=sample_features)
    # Puede ser 503 si modelo no está cargado, o 200 si está
    assert response.status_code in [200, 503]

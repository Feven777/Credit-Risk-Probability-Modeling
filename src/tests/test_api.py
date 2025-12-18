from fastapi.testclient import TestClient
from src.api.main import app
import pytest
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    """Test if the API starts correctly"""
    response = client.get("/health")
    assert response.status_code == 200

@patch("mlflow.pyfunc.load_model")
def test_predict_endpoint(mock_load):
    """Test the predict route without needing a real MLflow server"""
    payload = {
        "std_transaction_amount": 10.0,
        "Value": 100.0,
        "avg_transaction_amount": 20.0,
        "Amount": 100.0,
        "total_transaction_amount": 500.0,
        "transaction_count": 5,
        "transaction_day": 1,
        "transaction_hour": 10,
        "transaction_month": 1
    }
    # This just tests if the API can receive the Pydantic data correctly
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
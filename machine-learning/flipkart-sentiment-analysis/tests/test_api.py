import sys
from pathlib import Path
from fastapi.testclient import TestClient

#dynamically locate the project root folder and assign python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from api.main import app

client = TestClient(app)

# api status check
# 200 - request was successful
# 400 - bad request, request format is valid, but the business logic rejects it
# 404 - not found, endpoint doesn't exist
# 422 - unprocessable entity, FastAPI couldn't validate the incoming request body

#test root() endpoint
def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert "message" in data

#test health() endpoint
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["model_loaded"] is True

#test post() endpoint
def test_predict_positive_sentiment():
    response = client.post(
        "/predict",
        json={
            "text": "The service was excellent and very professional."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "sentiment" in data
    assert "confidence" in data
    assert "latency_ms" in data

def test_predict_negative_sentiment():
    response = client.post(
        "/predict",
        json={
            "text": "The product was very expensive and low quality."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "sentiment" in data
    assert "confidence" in data

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code in [400, 422]
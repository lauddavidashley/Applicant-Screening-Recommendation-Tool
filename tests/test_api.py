from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    response = client.post("/predict", json={"text": "experienced python developer with machine learning skills"})
    assert response.status_code == 200
    data = response.json()
    assert "top_matches" in data
    assert len(data["top_matches"]) > 0


def test_predict_rejects_short_input():
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422
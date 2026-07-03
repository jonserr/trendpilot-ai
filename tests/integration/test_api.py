from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_model_metadata_endpoint():
    response = client.get("/model-metadata")
    # 200 once a model has been trained, 404 on a fresh checkout
    assert response.status_code in (200, 404)


def test_prediction_history_disabled_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    response = client.get("/prediction-history")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["records"] == []

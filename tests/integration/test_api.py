from importlib.metadata import version

from fastapi.testclient import TestClient

from api.main import app
from src.config import get_config

client = TestClient(app)


VALID_PAYLOAD = {
    "features": {
        "region": "WA",
        "location": "Seattle, WA",
        "inventory": 1500,
        "mortgage_rate": 6.5,
        "days_on_market": 30,
        "sale_to_list_ratio": 1.0,
        "new_listings": 300,
        "month": 6,
        "quarter": 2,
        "year": 2026,
        "median_sale_price_lag_1": 800000.0,
        "median_sale_price_lag_2": 795000.0,
        "median_sale_price_lag_3": 790000.0,
        "median_sale_price_rolling_mean_3": 795000.0,
        "median_sale_price_rolling_mean_6": 780000.0,
    }
}


def test_payload_matches_active_config():
    """Keep this test payload honest against the dataset config."""
    cfg = get_config()
    assert set(VALID_PAYLOAD["features"]) == set(cfg.client_feature_cols)


def test_openapi_reports_package_version():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == version("trendpilot-ai")


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


def test_dataset_schema_endpoint():
    response = client.get("/dataset-schema")
    assert response.status_code == 200
    body = response.json()
    cfg = get_config()
    assert body["target_column"] == cfg.target_col
    assert body["entity_column"] == cfg.entity_col
    assert set(body["client_feature_columns"]) == set(cfg.client_feature_cols)


def test_predict_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)
    # 503 if the model has not been trained yet, 200 once it has
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["entity_id"] == "Seattle, WA"
        assert "predicted_value" in body


def test_predict_rejects_missing_features():
    cfg = get_config()
    payload = {
        "features": {k: v for k, v in VALID_PAYLOAD["features"].items() if k != cfg.entity_col}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert cfg.entity_col in str(response.json()["detail"])


def test_predict_rejects_non_numeric_feature():
    payload = {"features": {**VALID_PAYLOAD["features"], "inventory": "lots"}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_bad_month():
    payload = {"features": {**VALID_PAYLOAD["features"], "month": 13}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

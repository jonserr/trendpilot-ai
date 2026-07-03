from src.db.prediction_logger import fetch_history, is_enabled, log_prediction

RECORD = dict(
    endpoint="/predict",
    model_name="trend_model.keras",
    model_version="test-1",
    entity_id="Seattle, WA",
    entity_context={"region": "WA", "location": "Seattle, WA"},
    horizon=3,
    request_payload={"inventory": 1500},
    scenario_adjustments={"mortgage_rate": 6.5},
    prediction={"predicted_value": 412345.67},
    feature_columns=["inventory", "mortgage_rate"],
    warnings=["Applied scenario adjustment for mortgage_rate: 6.5"],
    duration_ms=12.5,
)


def test_disabled_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert is_enabled() is False
    assert log_prediction(**RECORD) is False
    assert fetch_history() == []


def test_log_and_fetch_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/history.db")
    assert is_enabled() is True

    assert log_prediction(**RECORD) is True
    records = fetch_history(limit=10)

    assert len(records) == 1
    record = records[0]
    assert record["entity_id"] == "Seattle, WA"
    assert record["entity_context"]["region"] == "WA"
    assert record["horizon"] == 3
    assert record["prediction"]["predicted_value"] == 412345.67
    assert record["scenario_adjustments"] == {"mortgage_rate": 6.5}
    assert record["warnings"]
    assert record["duration_ms"] == 12.5
    assert record["created_at"] is not None


def test_fetch_filters_by_entity(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/history.db")
    log_prediction(**RECORD)
    log_prediction(**{**RECORD, "entity_id": "Portland, OR"})

    assert len(fetch_history(entity_id="Portland, OR")) == 1
    assert len(fetch_history(entity_id="Seattle, WA")) == 1
    assert len(fetch_history()) == 2


def test_logging_fails_soft_on_bad_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://nobody:wrong@127.0.0.1:1/none")
    assert log_prediction(**RECORD) is False
    assert fetch_history() == []

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.services import model_service
from api.services.model_service import ModelService
from src.config import get_config
from src.modeling import train
from src.modeling.artifacts import resolve_bundle
from tests.fixtures.synthetic import generate_data


def test_offline_training_publishes_reloadable_bundle(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    data_path = tmp_path / "data.csv"
    generate_data(years=3, seed=7, geography={"AA": ["City One, AA", "City Two, AA"]}).to_csv(
        data_path, index=False
    )
    root = tmp_path / "models"
    monkeypatch.setattr(train, "MODELS_DIR", root)
    monkeypatch.setattr(model_service, "MODELS_DIR", root)
    monkeypatch.setattr(train, "TUNING_DIR", tmp_path / "tuning")
    monkeypatch.setattr(train, "ensure_dirs", lambda: None)
    cfg = replace(get_config(), data_file=str(data_path))
    monkeypatch.setattr(train, "get_config", lambda: cfg)
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("BATCH_SIZE", "16")
    assert TestClient(app).get("/ready").status_code == 503
    train.run_training_pipeline()
    bundle = resolve_bundle(root)
    assert bundle.validate()["model_version"] == bundle.bundle_id
    assert not (root / "saved/trend_model.keras").exists()
    assert not list((root / "bundles").glob(".staging-*"))
    ModelService._reload_if_stale()
    assert TestClient(app).get("/ready").status_code == 200
    sample = train.prepare_training_data()["validation_sample"]
    result = ModelService.get_instance().predict(sample)
    assert len(result) == len(sample) == 1

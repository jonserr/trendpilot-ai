import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.services import model_service
from api.services.model_service import ModelService
from tests.unit.test_artifacts import publish


class FakeForecaster:
    def __init__(self, model_path, preprocessor_path):
        self.value = model_path.read_text()
        self.preprocessor = SimpleNamespace(
            feature_names_in_=["x"], get_feature_names_out=lambda: ["x"]
        )
        self.model = SimpleNamespace(input_shape=(None, 1))

    def load(self):
        if self.value == "broken":
            raise ValueError("invalid model")
        return True

    def predict(self, df):
        return [float(self.value)]


@pytest.fixture(autouse=True)
def isolate_service(tmp_path, monkeypatch):
    monkeypatch.setattr(model_service, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(model_service, "Forecaster", FakeForecaster)
    return tmp_path


def test_cold_boot_is_live_but_not_ready_even_with_auth_enabled(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/dataset-schema").status_code == 401
    assert ModelService.get_instance().is_loaded is False


def test_valid_bundle_becomes_ready_and_reports_version(tmp_path):
    bundle = publish(tmp_path)
    ModelService._reload_if_stale()
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == bundle.bundle_id
    assert ModelService.get_instance().predict(None) == [1.0]


def test_failed_initial_load_stays_live_but_unready(tmp_path):
    publish(tmp_path, "broken")
    ModelService._reload_if_stale()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert ModelService.status()["reload_failed"] is True


def test_failed_replacement_preserves_last_working_snapshot(tmp_path):
    first = publish(tmp_path)
    ModelService._reload_if_stale()
    snapshot = ModelService.get_instance()
    publish(tmp_path, "broken")
    ModelService._reload_if_stale()
    assert ModelService.get_instance() is snapshot
    assert snapshot.predict(None) == [1.0]
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == first.bundle_id
    assert ModelService.status()["reload_failed"] is True


def test_pointer_corruption_does_not_evict_working_model(tmp_path):
    publish(tmp_path)
    ModelService._reload_if_stale()
    snapshot = ModelService.get_instance()
    (tmp_path / "current.json").write_text("bad-json")
    ModelService._reload_if_stale()
    assert ModelService.get_instance() is snapshot
    assert TestClient(app).get("/ready").status_code == 200


def test_inflight_requests_keep_one_bundle_during_reload(tmp_path, monkeypatch):
    publish(tmp_path)
    ModelService._reload_if_stale()
    snapshot = ModelService.get_instance()
    second = publish(tmp_path, "2")
    started, release = threading.Event(), threading.Event()

    class SlowForecaster(FakeForecaster):
        def load(self):
            started.set()
            assert release.wait(5)
            return super().load()

    monkeypatch.setattr(model_service, "Forecaster", SlowForecaster)
    ModelService._loading = True
    thread = threading.Thread(target=ModelService._reload_if_stale)
    thread.start()
    try:
        assert started.wait(5)
        assert ModelService.get_instance() is snapshot
        assert snapshot.predict(None) == [1.0]
        assert TestClient(app).get("/ready").status_code == 200
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert ModelService.get_instance().metadata["model_version"] == second.bundle_id
    assert ModelService.get_instance().predict(None) == [2.0]
    # A request holding its old dependency still sees the old metadata and model.
    assert snapshot.predict(None) == [1.0]
    assert snapshot.metadata["model_version"] != second.bundle_id


def test_metadata_preprocessor_mismatch_is_rejected(tmp_path, monkeypatch):
    class MismatchedForecaster(FakeForecaster):
        def load(self):
            self.preprocessor.feature_names_in_ = ["wrong-column"]
            return True

    monkeypatch.setattr(model_service, "Forecaster", MismatchedForecaster)
    publish(tmp_path)
    ModelService._reload_if_stale()
    assert TestClient(app).get("/ready").status_code == 503

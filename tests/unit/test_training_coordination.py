import pytest

from src.data import ingest
from src.modeling import train
from src.modeling.artifacts import TrainingInProgressError, training_lock
from tests.unit.test_artifacts import publish


def test_refresh_and_training_hold_one_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    events = []

    def assert_locked():
        with pytest.raises(TrainingInProgressError), training_lock(tmp_path):
            pass

    def refresh(args):
        assert_locked()
        assert args.level == "state" and args.max_entities is None
        events.append("refresh")

    def fit(bundle):
        assert_locked()
        assert bundle.directory.name.startswith(".staging-")
        events.append("train")
        return bundle

    monkeypatch.setattr(ingest, "run_ingestion", refresh)
    monkeypatch.setattr(train, "_train_bundle", fit)
    train.run_training_pipeline(refresh_data=True, level="state", max_entities=0)
    assert events == ["refresh", "train"]
    with training_lock(tmp_path):
        pass


def test_refresh_failure_preserves_active_pointer_and_releases_lock(tmp_path, monkeypatch):
    original = publish(tmp_path)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)

    def fail_refresh(args):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(ingest, "run_ingestion", fail_refresh)
    monkeypatch.setattr(train, "_train_bundle", lambda bundle: pytest.fail("must not train"))
    with pytest.raises(RuntimeError, match="refresh failed"):
        train.run_training_pipeline(refresh_data=True)
    assert original.bundle_id in (tmp_path / "current.json").read_text()
    with training_lock(tmp_path):
        pass

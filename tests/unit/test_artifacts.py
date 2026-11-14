import json
import subprocess
import sys

import pytest

from src.modeling.artifacts import (
    BundleError,
    publish_bundle,
    resolve_bundle,
    stage_bundle,
    training_lock,
)


def fill_bundle(bundle, value="1"):
    bundle.model_path.write_text(value)
    bundle.preprocessor_path.write_text("preprocessor")
    bundle.metadata_path.write_text(
        json.dumps({"model_version": bundle.bundle_id, "feature_columns": ["x"]})
    )


def publish(root, value="1"):
    with training_lock(root), stage_bundle(root) as bundle:
        fill_bundle(bundle, value)
        return publish_bundle(bundle)


def test_staging_is_invisible_until_complete_atomic_promotion(tmp_path):
    old = publish(tmp_path, "1")
    with training_lock(tmp_path), stage_bundle(tmp_path) as staged:
        staged.model_path.write_text("2")
        assert resolve_bundle(tmp_path) == old
        with pytest.raises(BundleError, match="Incomplete"):
            publish_bundle(staged)
        fill_bundle(staged, "2")
        new = publish_bundle(staged)
    assert resolve_bundle(tmp_path) == new
    assert old.model_path.read_text() == "1"
    assert new.model_path.read_text() == "2"
    assert new.validate()["model_version"] == new.bundle_id


def test_interrupted_training_preserves_active_bundle_and_cleans_stage(tmp_path):
    old = publish(tmp_path)
    with pytest.raises(RuntimeError), training_lock(tmp_path), stage_bundle(tmp_path) as staged:
        fill_bundle(staged)
        raise RuntimeError("training interrupted")
    assert resolve_bundle(tmp_path) == old
    assert not list((tmp_path / "bundles").glob(".staging-*"))


def test_failed_pointer_replacement_keeps_old_bundle(tmp_path, monkeypatch):
    from src.modeling import artifacts

    old = publish(tmp_path)

    def fail_replace(*args):
        raise OSError("simulated promotion interruption")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError), training_lock(tmp_path), stage_bundle(tmp_path) as staged:
        fill_bundle(staged, "2")
        publish_bundle(staged)
    assert resolve_bundle(tmp_path) == old
    assert not list(tmp_path.glob(".current-*.json"))


@pytest.mark.parametrize(
    "name", ["trend_model.keras", "preprocessor.joblib", "model_metadata.json"]
)
def test_each_artifact_is_checked_before_loading(tmp_path, name):
    bundle = publish(tmp_path)
    (bundle.directory / name).write_text("corrupt")
    with pytest.raises(BundleError, match="checksum"):
        bundle.validate()


@pytest.mark.parametrize("pointer", ["{", '{"format_version":1,"bundle_id":"../../other"}', "[]"])
def test_invalid_pointer_never_falls_back_to_legacy(tmp_path, pointer):
    (tmp_path / "current.json").write_text(pointer)
    with pytest.raises(BundleError):
        resolve_bundle(tmp_path)


def test_legacy_artifacts_remain_readable_without_a_pointer(tmp_path):
    (tmp_path / "saved").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "saved/trend_model.keras").write_text("legacy-model")
    (tmp_path / "metadata/preprocessor.joblib").write_text("legacy-preprocessor")
    (tmp_path / "metadata/model_metadata.json").write_text(
        json.dumps({"model_version": "existing-version", "feature_columns": ["x"]})
    )
    bundle = resolve_bundle(tmp_path)
    assert bundle.legacy
    assert bundle.validate()["model_version"] == "existing-version"
    new = publish(tmp_path)
    assert resolve_bundle(tmp_path) == new


def test_writer_lock_excludes_another_process_and_releases_after_exit(tmp_path):
    code = """
from pathlib import Path
import sys
from src.modeling.artifacts import training_lock, TrainingInProgressError
try:
    with training_lock(Path(sys.argv[1])):
        pass
except TrainingInProgressError:
    sys.exit(2)
"""
    command = [sys.executable, "-c", code, str(tmp_path)]
    with training_lock(tmp_path):
        assert subprocess.run(command, check=False, timeout=10).returncode == 2
    assert subprocess.run(command, check=False, timeout=10).returncode == 0
    assert (tmp_path / ".training.lock").is_file()


def test_writer_lock_is_released_when_owner_is_killed(tmp_path):
    code = """
from pathlib import Path
import sys
from src.modeling.artifacts import training_lock
with training_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
    finally:
        process.kill()
        process.wait(timeout=10)
    with training_lock(tmp_path):
        pass

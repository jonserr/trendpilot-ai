import os
from pathlib import Path
import stat
import sys

import pytest

from scripts import api_key


@pytest.fixture
def key_path(tmp_path, monkeypatch):
    path = tmp_path / "credentials" / "api-key"
    monkeypatch.setenv("TRENDPILOT_API_KEY_FILE", str(path))
    return path


def test_init_is_private_random_idempotent_and_does_not_print_value(key_path, capsys):
    assert api_key.main(["init"]) == 0
    value = key_path.read_text().strip()
    assert len(value) == 43  # token_urlsafe(32): 256 bits of entropy
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_path.parent.stat().st_mode) == 0o700
    assert api_key.main(["init"]) == 0
    assert api_key.main(["status"]) == 0
    assert key_path.read_text().strip() == value
    assert value not in capsys.readouterr().out


def test_rotation_replaces_file_and_changes_key(key_path, capsys):
    api_key.initialize(key_path)
    previous = key_path.read_text()
    previous_inode = key_path.stat().st_ino
    assert api_key.main(["rotate"]) == 0
    assert key_path.read_text() != previous
    assert key_path.stat().st_ino != previous_inode
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert previous.strip() not in capsys.readouterr().out


def test_key_storage_inside_checkout_is_rejected(monkeypatch):
    monkeypatch.setenv("TRENDPILOT_API_KEY_FILE", str(api_key.PROJECT_ROOT / "api-key"))
    with pytest.raises(ValueError, match="outside"):
        api_key.key_path()


def test_relative_key_storage_is_rejected(monkeypatch):
    monkeypatch.setenv("TRENDPILOT_API_KEY_FILE", "../api-key")
    with pytest.raises(ValueError, match="absolute"):
        api_key.key_path()


def test_default_location_is_project_specific_and_external(monkeypatch):
    monkeypatch.delenv("TRENDPILOT_API_KEY_FILE", raising=False)
    assert api_key.key_path() == Path.home() / ".config" / "trendpilot-ai" / "api-key"


@pytest.mark.parametrize("target", ["file", "directory"])
def test_overly_broad_permissions_are_rejected(key_path, target):
    api_key.initialize(key_path)
    (key_path if target == "file" else key_path.parent).chmod(0o644 if target == "file" else 0o755)
    with pytest.raises(ValueError, match="permissions"):
        api_key.check_key(key_path)


def test_init_never_overwrites_symlink_target(key_path, tmp_path):
    key_path.parent.mkdir(mode=0o700)
    target = tmp_path / "unrelated"
    target.write_text("preserve-me")
    key_path.symlink_to(target)
    with pytest.raises(OSError):
        api_key.initialize(key_path)
    assert target.read_text() == "preserve-me"


def test_shared_key_inode_is_rejected(key_path, tmp_path):
    api_key.initialize(key_path)
    os.link(key_path, tmp_path / "second-link")
    with pytest.raises(ValueError, match="unshared"):
        api_key.check_key(key_path)


def test_run_passes_only_path_and_preserves_exit_code(key_path, monkeypatch, capfd):
    api_key.initialize(key_path)
    monkeypatch.setenv("API_KEY", "do-not-inherit")
    code = (
        "import os, sys; "
        "assert 'API_KEY' not in os.environ; "
        "assert os.environ['API_KEY_REQUIRED'] == 'true'; "
        "assert os.environ['API_KEY_FILE'] == os.environ['TRENDPILOT_API_KEY_FILE']; "
        "sys.exit(7)"
    )
    assert api_key.main(["run", "--", sys.executable, "-c", code]) == 7
    assert "do-not-inherit" not in capfd.readouterr().out


def test_failed_rotation_preserves_existing_key(key_path, monkeypatch):
    api_key.initialize(key_path)
    previous = key_path.read_text()

    def fail_replace(*args):
        raise OSError("simulated filesystem error")

    monkeypatch.setattr(api_key.os, "replace", fail_replace)
    with pytest.raises(OSError):
        api_key.rotate(key_path)
    assert key_path.read_text() == previous
    assert list(key_path.parent.iterdir()) == [key_path]

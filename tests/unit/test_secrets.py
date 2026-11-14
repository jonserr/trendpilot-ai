import pytest

from src.utils.secrets import SecretConfigurationError, get_api_key


def test_auth_remains_optional_for_unconfigured_native_processes():
    assert get_api_key() is None


def test_file_key_and_rotation(tmp_path, monkeypatch):
    path = tmp_path / "api-key"
    path.write_text("first-key\n")
    monkeypatch.setenv("API_KEY_FILE", str(path))
    assert get_api_key() == "first-key"
    path.write_text("replacement-key\n")
    assert get_api_key() == "replacement-key"


@pytest.mark.parametrize(
    "content", ["", "\n", "\t", "contains spaces", "two\nlines", "é", "a" * 4097]
)
def test_invalid_key_file_fails_closed(tmp_path, monkeypatch, content):
    path = tmp_path / "api-key"
    path.write_text(content)
    monkeypatch.setenv("API_KEY_FILE", str(path))
    with pytest.raises(SecretConfigurationError):
        get_api_key()


def test_unreadable_configured_file_is_not_an_auth_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(SecretConfigurationError, match="cannot be read"):
        get_api_key()


def test_conflicting_sources_do_not_silently_select_a_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "do-not-disclose")
    monkeypatch.setenv("API_KEY_FILE", "/unused")
    with pytest.raises(SecretConfigurationError) as exc:
        get_api_key()
    assert "do-not-disclose" not in str(exc.value)


@pytest.mark.parametrize("value", ["true", "1"])
def test_required_auth_rejects_empty_key(monkeypatch, value):
    monkeypatch.setenv("API_KEY_REQUIRED", value)
    monkeypatch.setenv("API_KEY", "")
    with pytest.raises(SecretConfigurationError, match="required"):
        get_api_key()


def test_misspelled_requirement_cannot_disable_auth(monkeypatch):
    monkeypatch.setenv("API_KEY_REQUIRED", "tru")
    with pytest.raises(SecretConfigurationError):
        get_api_key()


def test_kubernetes_environment_key_from_file_accepts_terminal_newline(monkeypatch):
    monkeypatch.setenv("API_KEY", "external-key\n")
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    assert get_api_key() == "external-key"


def test_key_file_read_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "api-key"
    path.write_text("key" + "\n" * 5000)
    monkeypatch.setenv("API_KEY_FILE", str(path))
    with pytest.raises(SecretConfigurationError, match="too large"):
        get_api_key()

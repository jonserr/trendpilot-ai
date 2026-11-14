import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.utils.secrets import SecretConfigurationError


def test_file_auth_is_shared_by_api_and_dashboard(tmp_path, monkeypatch):
    from app.api_client import api_headers

    path = tmp_path / "api-key"
    path.write_text("file-key\n")
    monkeypatch.setenv("API_KEY_FILE", str(path))
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/dataset-schema").status_code == 401
        assert client.get("/dataset-schema", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/dataset-schema", headers=api_headers()).status_code == 200
        path.write_text("rotated-key\n")
        assert client.get("/dataset-schema", headers={"X-API-Key": "file-key"}).status_code == 401
        assert client.get("/dataset-schema", headers=api_headers()).status_code == 200


def test_empty_required_key_prevents_startup(monkeypatch):
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    monkeypatch.setenv("API_KEY", "")
    with pytest.raises(SecretConfigurationError):
        with TestClient(app):
            pass


def test_missing_file_after_startup_fails_closed_and_redacts_path(tmp_path, monkeypatch):
    path = tmp_path / "private-key-location"
    path.write_text("do-not-disclose\n")
    monkeypatch.setenv("API_KEY_FILE", str(path))
    with TestClient(app) as client:
        path.unlink()
        for endpoint in ("/dataset-schema", "/health"):
            response = client.get(endpoint)
            assert response.status_code == 503
            assert response.json() == {"detail": "API authentication unavailable"}


def test_environment_key_is_still_supported(monkeypatch):
    monkeypatch.setenv("API_KEY", "external-secret")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/dataset-schema").status_code == 401
        assert (
            client.get("/dataset-schema", headers={"X-API-Key": "external-secret"}).status_code
            == 200
        )

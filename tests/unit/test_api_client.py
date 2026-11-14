from app.api_client import api_headers


def test_api_headers_empty_without_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert api_headers() == {}


def test_api_headers_sends_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-123")
    assert api_headers() == {"X-API-Key": "secret-123"}

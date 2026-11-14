"""Keep tests independent of the developer's real credential configuration."""

import pytest


@pytest.fixture(autouse=True)
def isolate_auth_environment(monkeypatch):
    for name in ("API_KEY", "API_KEY_FILE", "API_KEY_REQUIRED"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolate_model_service(monkeypatch):
    """Unit/API tests must not start threads loading a developer's real model."""
    from api.services.model_service import ModelService

    for name, value in {
        "_instance": None,
        "_initialized": False,
        "_loading": False,
        "_reload_failed": False,
        "_retry_after": 0.0,
    }.items():
        monkeypatch.setattr(ModelService, name, value)
    monkeypatch.setattr(ModelService, "ensure_fresh_async", lambda: None)
    monkeypatch.setattr(ModelService, "warm_up_async", lambda: None)

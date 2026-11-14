"""Read API credentials without coupling the application to a secret backend."""

import os
from pathlib import Path


class SecretConfigurationError(RuntimeError):
    """Invalid credential configuration. Messages must never include secret values."""


def get_api_key() -> str | None:
    """Read API_KEY_FILE (Compose/Kubernetes) or the legacy API_KEY environment value.

    A configured but unreadable/empty file is an error, never an auth opt-out.
    Re-read files so native processes can pick up a rotated credential.
    """
    required = os.environ.get("API_KEY_REQUIRED", "false").lower()
    if required not in {"true", "false", "1", "0"}:
        raise SecretConfigurationError("API_KEY_REQUIRED must be true, false, 1, or 0")

    # kubectl --from-file preserves a terminal newline when injecting an env value.
    key = os.environ.get("API_KEY", "").rstrip("\r\n")
    filename = os.environ.get("API_KEY_FILE", "")
    if key and filename:
        raise SecretConfigurationError("Configure only one of API_KEY and API_KEY_FILE")
    if filename:
        try:
            with Path(filename).expanduser().open(encoding="utf-8") as handle:
                raw = handle.read(4099)
        except (OSError, UnicodeError):
            raise SecretConfigurationError("API_KEY_FILE cannot be read") from None
        if len(raw) > 4098:
            raise SecretConfigurationError("API_KEY_FILE is too large")
        key = raw.rstrip("\r\n")
        if not key:
            raise SecretConfigurationError("API_KEY_FILE must not be empty")

    # Keep credentials safe for HTTP headers and constant-time byte comparison.
    if key and (len(key) > 4096 or not key.isascii() or any(not 33 <= ord(c) <= 126 for c in key)):
        raise SecretConfigurationError("API key must contain printable ASCII without whitespace")
    if not key and required in {"true", "1"}:
        raise SecretConfigurationError("API key is required; configure API_KEY_FILE or API_KEY")
    return key or None

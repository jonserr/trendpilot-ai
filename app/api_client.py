"""Inference API connection settings shared by all dashboard pages.

``API_URL`` points at the FastAPI service. When a key is configured, every
request carries it in the ``X-API-Key`` header the API middleware expects.
"""

import os

from src.utils.secrets import get_api_key

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def api_headers() -> dict[str, str]:
    """Request headers for the inference API (empty when auth is disabled)."""
    api_key = get_api_key()
    return {"X-API-Key": api_key} if api_key else {}

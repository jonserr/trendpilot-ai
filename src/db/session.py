"""Engine management driven by the DATABASE_URL environment variable."""

import os
from functools import lru_cache


def get_database_url() -> str | None:
    """Return the configured database URL, or None when logging is disabled."""
    return os.environ.get("DATABASE_URL") or None


@lru_cache(maxsize=4)
def _engine_for(url: str):
    from sqlalchemy import create_engine

    return create_engine(url, pool_pre_ping=True)


def get_engine():
    """Return a cached SQLAlchemy engine, or None if no DATABASE_URL is set
    or the engine cannot be created."""
    url = get_database_url()
    if not url:
        return None
    try:
        return _engine_for(url)
    except Exception:
        return None

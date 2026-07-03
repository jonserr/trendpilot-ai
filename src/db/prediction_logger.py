"""Fail-soft prediction logging.

`log_prediction` and `fetch_history` never raise: inference must keep working
when DATABASE_URL is unset, the database is unreachable, or drivers are
missing. Failures are logged as warnings and swallowed.
"""

import logging

logger = logging.getLogger(__name__)

_initialized_engines: set = set()


def is_enabled() -> bool:
    from src.db.session import get_database_url

    return get_database_url() is not None


def _ensure_tables(engine) -> None:
    from src.db.models import Base

    key = str(engine.url)
    if key not in _initialized_engines:
        Base.metadata.create_all(engine)
        _initialized_engines.add(key)


def log_prediction(
    *,
    endpoint: str,
    model_name: str,
    model_version: str,
    entity_id: str,
    entity_context: dict | None,
    horizon: int,
    request_payload: dict,
    scenario_adjustments: dict | None,
    prediction: dict,
    feature_columns: list,
    warnings: list,
    duration_ms: float,
) -> bool:
    """Persist one prediction. Returns True on success, False otherwise."""
    try:
        from src.db.session import get_engine

        engine = get_engine()
        if engine is None:
            return False

        from sqlalchemy.orm import Session

        from src.db.models import PredictionRecord

        _ensure_tables(engine)
        with Session(engine) as session:
            session.add(
                PredictionRecord(
                    endpoint=endpoint,
                    model_name=model_name,
                    model_version=model_version,
                    entity_id=entity_id,
                    entity_context=entity_context,
                    horizon=horizon,
                    request_payload=request_payload,
                    scenario_adjustments=scenario_adjustments,
                    prediction=prediction,
                    feature_columns=feature_columns,
                    warnings=warnings,
                    duration_ms=duration_ms,
                )
            )
            session.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - logging must never break inference
        logger.warning("Prediction logging skipped: %s", exc)
        return False


def fetch_history(limit: int = 50, entity_id: str | None = None) -> list:
    """Return recent prediction records (newest first). Empty list on any failure."""
    try:
        from src.db.session import get_engine

        engine = get_engine()
        if engine is None:
            return []

        from sqlalchemy import desc, select
        from sqlalchemy.orm import Session

        from src.db.models import PredictionRecord

        _ensure_tables(engine)
        stmt = select(PredictionRecord)
        if entity_id:
            stmt = stmt.where(PredictionRecord.entity_id == entity_id)
        stmt = stmt.order_by(desc(PredictionRecord.created_at), desc(PredictionRecord.id))
        stmt = stmt.limit(max(1, min(int(limit), 500)))

        with Session(engine) as session:
            return [record.to_dict() for record in session.scalars(stmt)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetching prediction history failed: %s", exc)
        return []

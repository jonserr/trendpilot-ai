from typing import Optional

from fastapi import APIRouter, Query

from src.db.prediction_logger import fetch_history, is_enabled

router = APIRouter()


@router.get("/prediction-history", tags=["History"])
def prediction_history(
    limit: int = Query(50, ge=1, le=500),
    entity_id: Optional[str] = None,
):
    """Recent logged predictions. Requires DATABASE_URL; otherwise reports disabled."""
    if not is_enabled():
        return {
            "enabled": False,
            "count": 0,
            "records": [],
            "detail": "Prediction logging is disabled. Set DATABASE_URL to enable it.",
        }
    records = fetch_history(limit=limit, entity_id=entity_id)
    return {"enabled": True, "count": len(records), "records": records}

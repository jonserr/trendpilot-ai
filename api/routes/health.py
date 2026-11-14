from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.services.model_service import ModelService

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check():
    """Fast, non-blocking health check: never blocks on a model load, but
    kicks off a background reload when the active bundle changes."""
    ModelService.ensure_fresh_async()
    status = ModelService.status()
    return {
        "status": "ok",
        "model_file_exists": ModelService.model_file_exists(),
        "model_loaded": status["model_loaded"],
        "model_loading": status["loading"] or not status["initialized"],
        "metadata_exists": status["metadata_exists"],
        "model_version": status["model_version"],
        "reload_failed": status["reload_failed"],
    }


@router.get("/ready", tags=["System"])
def readiness_check():
    """Only validated inference snapshots are ready, including during a reload."""
    ModelService.ensure_fresh_async()
    status = ModelService.status()
    ready = status["model_loaded"] and status["metadata_exists"]
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "model_version": status["model_version"],
        },
    )

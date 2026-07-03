from fastapi import APIRouter

from api.services.model_service import ModelService
from src.utils.paths import SAVED_MODELS_DIR

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check():
    """Fast, non-blocking health check: never blocks on a model load, but
    kicks off a background (re)load when a new model file appears on disk."""
    ModelService.ensure_fresh_async()
    status = ModelService.status()
    return {
        "status": "ok",
        "model_file_exists": (SAVED_MODELS_DIR / "trend_model.keras").exists(),
        "model_loaded": status["model_loaded"],
        "model_loading": status["loading"] or not status["initialized"],
        "metadata_exists": status["metadata_exists"],
    }

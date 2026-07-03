import json
import logging
import threading

from src.modeling.predict import Forecaster
from src.utils.paths import METADATA_DIR, SAVED_MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = SAVED_MODELS_DIR / "trend_model.keras"


class ModelService:
    """Thread-safe lazy singleton around the trained model.

    Loading TensorFlow and the saved model can take several seconds, so the
    API warms the instance up in a background thread at startup
    (``warm_up_async``) and ``/health`` reads ``status()`` without ever
    triggering a blocking load.

    The service also watches the model file's mtime: when a newer model
    appears on disk (e.g. after training from the dashboard), the next
    health check triggers a background reload — no API restart needed.
    """

    _instance = None
    _lock = threading.Lock()
    _loading = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._loading = True
                    try:
                        cls._instance = cls()
                    finally:
                        cls._loading = False
        return cls._instance

    @staticmethod
    def _model_mtime() -> float | None:
        try:
            return MODEL_PATH.stat().st_mtime
        except OSError:
            return None

    @classmethod
    def _is_stale(cls) -> bool:
        """True when a model file exists that the current instance hasn't
        loaded (first load or a newer file written by training)."""
        mtime = cls._model_mtime()
        if mtime is None:
            return False
        instance = cls._instance
        if instance is None or not instance.is_loaded:
            return True
        return (instance.loaded_mtime or 0) < mtime

    @classmethod
    def _reload_if_stale(cls):
        with cls._lock:
            if not cls._is_stale():
                return
            cls._loading = True
            try:
                cls._instance = cls()
                logger.info("Model (re)loaded from %s.", MODEL_PATH)
            finally:
                cls._loading = False

    @classmethod
    def warm_up_async(cls) -> None:
        """Load (or reload) the model in the background so requests don't
        pay the TensorFlow import + model load cost."""
        if cls._loading:
            return

        def _load():
            try:
                if cls._instance is None:
                    cls.get_instance()
                else:
                    cls._reload_if_stale()
                logger.info("Model warm-up complete.")
            except Exception:  # noqa: BLE001 - warm-up must never crash the API
                logger.exception("Model warm-up failed; will retry on next check.")

        threading.Thread(target=_load, name="model-warm-up", daemon=True).start()

    @classmethod
    def ensure_fresh_async(cls) -> None:
        """Kick off a background reload when the on-disk model is newer than
        the loaded one. Cheap (a single stat) and non-blocking."""
        if cls._is_stale():
            cls.warm_up_async()

    @classmethod
    def status(cls) -> dict:
        """Non-blocking snapshot for health checks."""
        instance = cls._instance
        return {
            "initialized": instance is not None,
            "loading": cls._loading,
            "model_loaded": bool(instance.is_loaded) if instance else False,
            "metadata_exists": bool(instance.metadata) if instance else False,
        }

    def __init__(self):
        self.forecaster = Forecaster()
        self.is_loaded = self.forecaster.load()
        self.loaded_mtime = self._model_mtime() if self.is_loaded else None
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        metadata_path = METADATA_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def predict(self, df):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        return self.forecaster.predict(df)

    def get_metadata(self):
        return self.metadata

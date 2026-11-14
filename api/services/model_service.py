import logging
import threading
import time

from src.modeling.artifacts import BundleError, ModelBundle, resolve_bundle
from src.modeling.predict import Forecaster
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelService:
    """Load complete candidates off-thread and swap only after validation.

    Requests keep their service snapshot for their whole prediction. A failed
    reload leaves the last working model/preprocessor/metadata together in memory.
    """

    _instance = None
    _lock = threading.Lock()
    _loading = False
    _initialized = False
    _reload_failed = False
    _retry_after = 0.0
    RETRY_SECONDS = 5.0

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            instance = cls._instance
        cls.ensure_fresh_async()
        return instance

    @classmethod
    def _is_stale(cls) -> bool:
        try:
            bundle = resolve_bundle(MODELS_DIR)
        except (BundleError, OSError):
            return True
        if bundle is None:
            return False
        instance = cls._instance
        return instance is None or instance.loaded_revision != bundle.revision

    @classmethod
    def _reload_if_stale(cls):
        try:
            bundle = resolve_bundle(MODELS_DIR)
            instance = cls._instance
            if bundle is not None and (
                instance is None or instance.loaded_revision != bundle.revision
            ):
                candidate = cls(bundle)
                if not candidate.is_loaded:
                    raise BundleError("Candidate model could not be loaded")
                # Legacy files are mutable; refuse a mixed snapshot during migration.
                if bundle.legacy and resolve_bundle(MODELS_DIR) != bundle:
                    raise BundleError("Legacy artifacts changed while loading")
                with cls._lock:
                    cls._instance = candidate
                logger.info("Model bundle loaded: %s", bundle.bundle_id)
            with cls._lock:
                cls._reload_failed = False
        except Exception:
            logger.exception("Model reload failed; retaining the previous working bundle")
            with cls._lock:
                cls._reload_failed = True
        finally:
            with cls._lock:
                cls._initialized = True
                cls._loading = False
                cls._retry_after = time.monotonic() + cls.RETRY_SECONDS

    @classmethod
    def warm_up_async(cls) -> None:
        with cls._lock:
            if cls._loading or time.monotonic() < cls._retry_after:
                return
            cls._loading = True
        threading.Thread(target=cls._reload_if_stale, name="model-warm-up", daemon=True).start()

    @classmethod
    def ensure_fresh_async(cls) -> None:
        if not cls._initialized or cls._is_stale():
            cls.warm_up_async()

    @classmethod
    def status(cls) -> dict:
        with cls._lock:
            instance = cls._instance
            return {
                "initialized": cls._initialized,
                "loading": cls._loading,
                "model_loaded": bool(instance and instance.is_loaded),
                "metadata_exists": bool(instance and instance.metadata),
                "model_version": (
                    instance.metadata.get("model_version")
                    if instance and instance.metadata
                    else None
                ),
                "reload_failed": cls._reload_failed,
            }

    @staticmethod
    def model_file_exists() -> bool:
        try:
            bundle = resolve_bundle(MODELS_DIR)
            return bundle is not None and bundle.model_path.is_file()
        except (BundleError, OSError):
            return False

    def __init__(self, bundle: ModelBundle | None = None):
        self.is_loaded = False
        self.metadata = None
        self.forecaster = None
        self.loaded_revision = None
        if bundle is not None:
            metadata = bundle.validate()
            forecaster = Forecaster(bundle.model_path, bundle.preprocessor_path)
            if not forecaster.load():
                raise BundleError("Model or preprocessor could not be loaded")
            columns = list(forecaster.preprocessor.feature_names_in_)
            if columns != metadata["feature_columns"]:
                raise BundleError("Preprocessor columns do not match bundle metadata")
            width = len(forecaster.preprocessor.get_feature_names_out())
            if forecaster.model.input_shape[-1] != width:
                raise BundleError("Model input size does not match the preprocessor")
            self.forecaster = forecaster
            self.metadata = metadata
            self.loaded_revision = bundle.revision
            self.is_loaded = True

    def predict(self, df):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        return self.forecaster.predict(df)

    def get_metadata(self):
        return self.metadata

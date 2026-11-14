import logging
from pathlib import Path

import pandas as pd

from src.data.preprocess import load_preprocessor
from src.modeling.artifacts import resolve_bundle

logger = logging.getLogger(__name__)


class Forecaster:
    """Thin inference wrapper. TensorFlow is imported lazily so the API and
    tests can run (and fail gracefully) without a trained model."""

    def __init__(
        self,
        model_path: Path | None = None,
        preprocessor_path: Path | None = None,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None

    def load(self) -> bool:
        """Loads the model and preprocessor if they exist. Fails soft (the
        API reports model_loaded=False) when TensorFlow is unavailable."""
        if self.model_path is None and self.preprocessor_path is None:
            bundle = resolve_bundle()
            if bundle is None:
                return False
            bundle.validate()
            self.model_path = bundle.model_path
            self.preprocessor_path = bundle.preprocessor_path
        if self.model_path is None or self.preprocessor_path is None:
            raise ValueError("Model and preprocessor paths must be provided together")
        if self.model_path.exists() and self.preprocessor_path.exists():
            try:
                import tensorflow as tf
            except ImportError:
                logger.warning("TensorFlow is not installed; model not loaded.")
                return False

            self.model = tf.keras.models.load_model(str(self.model_path))
            self.preprocessor = load_preprocessor(str(self.preprocessor_path))
            return True
        return False

    def predict(self, df: pd.DataFrame) -> list[float]:
        """Runs inference on a dataframe of features."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded. Call load() first.")

        X_scaled = self.preprocessor.transform(df)
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        return [float(p) for p in predictions]

"""Training pipeline: data -> features -> time-aware split -> baseline + Keras model.

Hyperparameters come from (in increasing precedence):
defaults -> tuned hyperparameters saved by `make tune` -> environment variables
(EPOCHS, BATCH_SIZE, LEARNING_RATE) set e.g. by the dashboard training page.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_config  # noqa: E402
from src.data.features import engineer_features  # noqa: E402
from src.data.loaders import load_data  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    get_preprocessor,
    save_preprocessor,
    time_aware_split,
)
from src.modeling.artifacts import (  # noqa: E402
    ModelBundle,
    TrainingInProgressError,
    publish_bundle,
    stage_bundle,
    training_lock,
)
from src.modeling.predict import Forecaster  # noqa: E402
from src.modeling.baseline import BaselineModel  # noqa: E402
from src.modeling.evaluate import calculate_metrics  # noqa: E402
from src.modeling.keras_model import DEFAULT_HPARAMS, build_dense_forecaster  # noqa: E402
from src.utils.paths import (  # noqa: E402
    MODELS_DIR,
    TUNING_DIR,
    ensure_dirs,
)


def load_tuned_hyperparameters() -> dict | None:
    """Return hyperparameters saved by the optional KerasTuner workflow, if any."""
    path = TUNING_DIR / "best_hyperparameters.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def prepare_training_data():
    """Load, engineer, split, and preprocess the dataset. Shared with tuning."""
    cfg = get_config()
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {cfg.data_path}. Run 'make fetch-data' first, "
            "or place your own dataset there (see config/dataset.yaml)."
        )

    df = engineer_features(load_data(cfg=cfg), cfg=cfg)
    train_df, val_df, test_df = time_aware_split(df, date_col=cfg.date_col)

    feature_cols = [c for c in cfg.model_feature_cols if c in df.columns]
    categorical_features = [c for c in cfg.categorical_features if c in feature_cols]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    preprocessor = get_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    return {
        "cfg": cfg,
        "X_train": X_train,
        "y_train": train_df[cfg.target_col],
        "X_val": X_val,
        "y_val": val_df[cfg.target_col],
        "X_test": X_test,
        "y_test": test_df[cfg.target_col],
        "preprocessor": preprocessor,
        "validation_sample": test_df[feature_cols].head(1),
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }


def run_training_pipeline(
    *, refresh_data: bool = False, level: str = "metro", max_entities: int = 50
) -> ModelBundle:
    # Dashboard, CLI, scheduled and manually created Jobs all take the same lock.
    with training_lock(MODELS_DIR):
        if refresh_data:
            from src.data.ingest import parse_args, run_ingestion

            run_ingestion(parse_args(["--level", level, "--max-entities", str(max_entities)]))
        with stage_bundle(MODELS_DIR) as bundle:
            return _train_bundle(bundle)


def _train_bundle(bundle: ModelBundle) -> ModelBundle:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import load_model

    ensure_dirs()

    print("Loading and preparing data...")
    prepared = prepare_training_data()
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_val, y_val = prepared["X_val"], prepared["y_val"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    save_preprocessor(prepared["preprocessor"], bundle.preprocessor_path)

    # Resolve hyperparameters: defaults <- tuned <- env overrides
    tuned = load_tuned_hyperparameters()
    hparams = {**DEFAULT_HPARAMS, **(tuned or {})}
    if os.environ.get("LEARNING_RATE"):
        hparams["learning_rate"] = float(os.environ["LEARNING_RATE"])
    epochs = int(os.environ.get("EPOCHS", 100))
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    print(f"Hyperparameters ({'tuned' if tuned else 'default'}): {hparams}")

    print("Training Baseline Model...")
    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    baseline_metrics = calculate_metrics(y_test, baseline.predict(X_test))
    print("Baseline Metrics:", baseline_metrics)

    print("Training Keras Model...")
    model = build_dense_forecaster(X_train.shape[1], hparams)
    model_path = str(bundle.model_path)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating Keras Model...")
    model = load_model(model_path)  # best checkpoint
    keras_metrics = calculate_metrics(y_test, model.predict(X_test).flatten())
    print("Keras Metrics:", keras_metrics)

    cfg = prepared["cfg"]
    metadata = {
        "model_name": "trend_model.keras",
        "model_type": "Dense Neural Network",
        "model_version": bundle.bundle_id,
        "training_date": datetime.now().isoformat(),
        "hyperparameters": hparams,
        "used_tuned_hyperparameters": tuned is not None,
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_name": cfg.name,
        "entity_column": cfg.entity_col,
        "date_column": cfg.date_col,
        "feature_columns": prepared["feature_cols"],
        "numeric_features": prepared["numeric_features"],
        "categorical_features": prepared["categorical_features"],
        "target_column": cfg.target_col,
        "target_unit": cfg.target_unit,
        "split_sizes": prepared["split_sizes"],
        "metrics": keras_metrics,
        "baseline_metrics": baseline_metrics,
    }
    with bundle.metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=4, allow_nan=False)

    # Probe the serialized model/preprocessor together before making them visible.
    import math

    candidate = Forecaster(bundle.model_path, bundle.preprocessor_path)
    if not candidate.load():
        raise RuntimeError("Trained bundle could not be reloaded")
    probe = candidate.predict(prepared["validation_sample"])
    if not probe or not all(math.isfinite(value) for value in probe):
        raise RuntimeError("Trained bundle failed prediction validation")
    published = publish_bundle(bundle)
    print(f"Training complete! Published model bundle {published.bundle_id}.")
    return published


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--level", default="metro", choices=["metro", "state", "county", "city", "zip"]
    )
    parser.add_argument("--max-entities", type=int, default=50)
    args = parser.parse_args()
    try:
        run_training_pipeline(
            refresh_data=args.refresh_data, level=args.level, max_entities=args.max_entities
        )
    except TrainingInProgressError as exc:
        parser.exit(2, f"{exc}\n")

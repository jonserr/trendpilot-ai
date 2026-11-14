"""Optional KerasTuner hyperparameter search.

Runs a small, CPU-friendly RandomSearch over layer count, units, dropout,
learning rate, and L2 regularization. Results are written to
models/metadata/tuning/ and automatically picked up by the next `make train`.

The rest of the application works without this ever being run.
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.keras_model import build_dense_forecaster  # noqa: E402
from src.modeling.train import prepare_training_data  # noqa: E402
from src.utils.paths import TUNING_DIR, ensure_dirs  # noqa: E402

SEARCH_SPACE = {
    "num_layers": [1, 2, 3],
    "units": [32, 64, 128, 256],
    "dropout": {"min": 0.0, "max": 0.4, "step": 0.1},
    "learning_rate": [0.01, 0.001, 0.0005, 0.0001],
    "l2": [0.0, 0.0001, 0.001],
}


def _hp_to_dict(hp_values: dict) -> dict:
    """Convert flat KerasTuner values into the dict build_dense_forecaster expects."""
    hidden_layers = [hp_values[f"units_{i}"] for i in range(hp_values["num_layers"])]
    return {
        "hidden_layers": hidden_layers,
        "dropout": hp_values["dropout"],
        "learning_rate": hp_values["learning_rate"],
        "l2": hp_values["l2"],
    }


def _build_from_tuner(hp, input_dim: int):
    num_layers = hp.Int("num_layers", 1, 3)
    values = {"num_layers": num_layers}
    for i in range(3):
        # Declare all unit choices so KerasTuner tracks a stable space,
        # only the first `num_layers` are used.
        values[f"units_{i}"] = hp.Choice(f"units_{i}", SEARCH_SPACE["units"])
    values["dropout"] = hp.Float("dropout", 0.0, 0.4, step=0.1)
    values["learning_rate"] = hp.Choice("learning_rate", SEARCH_SPACE["learning_rate"])
    values["l2"] = hp.Choice("l2", SEARCH_SPACE["l2"])
    return build_dense_forecaster(input_dim, _hp_to_dict(values))


def run_tuning(max_trials: int | None = None, epochs: int | None = None):
    try:
        import keras_tuner as kt
    except ImportError as exc:
        raise SystemExit(
            "keras-tuner is not installed. Install the optional tuning extra:\n"
            "  uv sync --locked --extra tuning\n"
            "or rebuild the Docker image (it includes the extra by default)."
        ) from exc

    from tensorflow.keras.callbacks import EarlyStopping

    ensure_dirs()
    max_trials = max_trials or int(os.environ.get("TUNER_MAX_TRIALS", 8))
    epochs = epochs or int(os.environ.get("TUNER_EPOCHS", 40))

    print("Preparing data for tuning...")
    prepared = prepare_training_data()
    input_dim = prepared["X_train"].shape[1]

    tuner = kt.RandomSearch(
        lambda hp: _build_from_tuner(hp, input_dim),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=str(TUNING_DIR / "keras_tuner"),
        project_name="dense_forecaster",
        overwrite=True,
        seed=42,
    )
    tuner.search_space_summary()

    print(f"Running RandomSearch: {max_trials} trials x up to {epochs} epochs (CPU-friendly)...")
    tuner.search(
        prepared["X_train"],
        prepared["y_train"],
        validation_data=(prepared["X_val"], prepared["y_val"]),
        epochs=epochs,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_hparams = _hp_to_dict(best_hp.values)

    with open(TUNING_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_hparams, f, indent=4)

    summary = {
        "tuner": "RandomSearch",
        "objective": "val_loss",
        "max_trials": max_trials,
        "epochs_per_trial": epochs,
        "search_space": SEARCH_SPACE,
        "best_val_loss": best_trial.score,
        "best_hyperparameters": best_hparams,
        "raw_best_values": best_hp.values,
        "tuning_date": datetime.now().isoformat(),
        "feature_columns": prepared["feature_cols"],
    }
    with open(TUNING_DIR / "tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Best hyperparameters: {best_hparams}")
    print(f"Saved to {TUNING_DIR}. Run 'make train' to train with them.")


if __name__ == "__main__":
    run_tuning()

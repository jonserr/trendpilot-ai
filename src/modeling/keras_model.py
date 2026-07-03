"""Keras model factory.

`build_dense_forecaster` accepts an optional hyperparameter dict so the same
factory serves the default training pipeline and the KerasTuner workflow.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

DEFAULT_HPARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "l2": 0.0,
}


def build_dense_forecaster(input_dim: int, hparams: dict | None = None) -> models.Model:
    """Builds a dense neural network forecaster from a hyperparameter dict."""
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    l2 = float(hp["l2"])
    kernel_regularizer = regularizers.l2(l2) if l2 > 0 else None

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for units in hp["hidden_layers"]:
        x = layers.Dense(int(units), activation="relu", kernel_regularizer=kernel_regularizer)(x)
        if float(hp["dropout"]) > 0:
            x = layers.Dropout(float(hp["dropout"]))(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="dense_forecaster")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(hp["learning_rate"])),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model

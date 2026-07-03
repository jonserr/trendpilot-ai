"""Dataset configuration loading and validation.

The pipeline is dataset-agnostic: feature engineering, preprocessing,
training, the inference API, and the dashboard all read the schema of the
active dataset from a single YAML config (``config/dataset.yaml`` by default,
overridable via the ``TRENDPILOT_CONFIG`` environment variable).

A dataset is any CSV with one row per (entity, date) observation at a regular
frequency, a numeric target column, and optional numeric/categorical driver
features.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from src.utils.paths import RAW_DATA_DIR, ROOT_DIR

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "dataset.yaml"

CALENDAR_COLS = ("month", "quarter", "year")
SEASONAL_COLS = ("month_sin", "month_cos")


class DataValidationError(ValueError):
    """Raised when a dataset does not match the active configuration."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_file: str
    date_col: str
    target_col: str
    entity_col: str
    entity_hierarchy: tuple = ()
    display_cols: tuple = ()
    categorical_features: tuple = ()
    numeric_features: tuple = ()
    lags: tuple = (1, 2, 3)
    rolling_windows: tuple = (3, 6)
    target_unit: str = ""
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.entity_hierarchy:
            object.__setattr__(self, "entity_hierarchy", (self.entity_col,))
        if self.entity_hierarchy[-1] != self.entity_col:
            raise DataValidationError(
                f"entity_hierarchy must end with entity_col ('{self.entity_col}'), "
                f"got {list(self.entity_hierarchy)}"
            )
        if any(lag < 1 for lag in self.lags):
            raise DataValidationError("All lags must be >= 1 to prevent target leakage.")
        if any(w < 1 for w in self.rolling_windows):
            raise DataValidationError("All rolling windows must be >= 1.")

    # ---- Derived column groups -------------------------------------------------

    @property
    def data_path(self) -> Path:
        return RAW_DATA_DIR / self.data_file

    @property
    def lag_cols(self) -> list:
        return [f"{self.target_col}_lag_{lag}" for lag in self.lags]

    @property
    def rolling_cols(self) -> list:
        return [f"{self.target_col}_rolling_mean_{w}" for w in self.rolling_windows]

    @property
    def raw_required_cols(self) -> list:
        """Columns a raw dataset must contain."""
        cols = [self.date_col, self.target_col]
        for col in (*self.entity_hierarchy, *self.categorical_features, *self.numeric_features):
            if col not in cols:
                cols.append(col)
        return cols

    @property
    def model_feature_cols(self) -> list:
        """Every column the trained model consumes, in canonical order."""
        cols = []
        for col in (
            *self.categorical_features,
            *self.numeric_features,
            *CALENDAR_COLS,
            *SEASONAL_COLS,
            *self.lag_cols,
            *self.rolling_cols,
        ):
            if col not in cols:
                cols.append(col)
        return cols

    @property
    def client_feature_cols(self) -> list:
        """Features an API client must provide (seasonal terms are derived
        server-side from ``month``)."""
        return [c for c in self.model_feature_cols if c not in SEASONAL_COLS]


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def load_dataset_config(path: str | Path | None = None) -> DatasetConfig:
    """Load the dataset config from ``path``, the ``TRENDPILOT_CONFIG``
    environment variable, or the default ``config/dataset.yaml``."""
    resolved = Path(path or os.environ.get("TRENDPILOT_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset config not found at {resolved}")

    with open(resolved) as f:
        raw = yaml.safe_load(f) or {}

    known = {
        "name",
        "data_file",
        "date_col",
        "target_col",
        "entity_col",
        "entity_hierarchy",
        "display_cols",
        "categorical_features",
        "numeric_features",
        "lags",
        "rolling_windows",
        "target_unit",
    }
    missing = {"name", "data_file", "date_col", "target_col", "entity_col"} - raw.keys()
    if missing:
        raise DataValidationError(f"Dataset config {resolved} is missing keys: {sorted(missing)}")

    extras = {k: v for k, v in raw.items() if k not in known}
    return DatasetConfig(
        name=raw["name"],
        data_file=raw["data_file"],
        date_col=raw["date_col"],
        target_col=raw["target_col"],
        entity_col=raw["entity_col"],
        entity_hierarchy=_as_tuple(raw.get("entity_hierarchy")),
        display_cols=_as_tuple(raw.get("display_cols")),
        categorical_features=_as_tuple(raw.get("categorical_features")),
        numeric_features=_as_tuple(raw.get("numeric_features")),
        lags=_as_tuple(raw.get("lags")) or (1, 2, 3),
        rolling_windows=_as_tuple(raw.get("rolling_windows")) or (3, 6),
        target_unit=raw.get("target_unit", "") or "",
        extras=extras,
    )


_config_cache: dict = {}


def get_config(path: str | Path | None = None) -> DatasetConfig:
    """Cached accessor for the active dataset config."""
    key = str(path or os.environ.get("TRENDPILOT_CONFIG") or DEFAULT_CONFIG_PATH)
    if key not in _config_cache:
        _config_cache[key] = load_dataset_config(path)
    return _config_cache[key]


def validate_dataframe(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Validate a raw dataset against the config. Returns the dataframe with
    the date column parsed; raises DataValidationError with actionable detail."""
    missing = [c for c in cfg.raw_required_cols if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns {missing}. "
            f"Expected schema (from dataset config '{cfg.name}'): {cfg.raw_required_cols}"
        )

    df = df.copy()
    try:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Column '{cfg.date_col}' is not parseable as dates: {exc}")

    for col in (cfg.target_col, *cfg.numeric_features):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric, got {df[col].dtype}.")

    if df[cfg.target_col].isna().all():
        raise DataValidationError(f"Target column '{cfg.target_col}' contains no values.")

    duplicated = df.duplicated(subset=[cfg.entity_col, cfg.date_col])
    if duplicated.any():
        raise DataValidationError(
            f"Dataset contains {int(duplicated.sum())} duplicate "
            f"({cfg.entity_col}, {cfg.date_col}) rows; one row per entity per period is required."
        )
    return df

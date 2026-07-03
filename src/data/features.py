"""Leakage-safe feature engineering for time-series forecasting.

All lag and rolling features are grouped by the configured entity column so
history never crosses series boundaries, and they use only past observations:
lags shift by >= 1 and rolling windows are computed on shift(1), so the
feature value at time t never includes the target at time t or later.

Column names come from the active dataset config (see ``src/config.py``),
which makes this module dataset-agnostic.
"""

import numpy as np
import pandas as pd

from src.config import DatasetConfig, get_config


def resolve_group_col(df: pd.DataFrame, cfg: DatasetConfig | None = None) -> str:
    """Group features at the finest available level of the entity hierarchy."""
    cfg = cfg or get_config()
    for col in reversed(cfg.entity_hierarchy):
        if col in df.columns:
            return col
    raise KeyError(
        f"None of the configured entity columns {list(cfg.entity_hierarchy)} "
        "are present in the dataframe."
    )


def seasonality_from_month(month):
    """Return (sin, cos) cyclical encodings for a 1-12 month value."""
    angle = 2.0 * np.pi * (np.asarray(month, dtype=float) - 1.0) / 12.0
    return np.sin(angle), np.cos(angle)


def create_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    return df


def add_seasonality_terms(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    df = df.copy()
    sin_term, cos_term = seasonality_from_month(df[month_col])
    df["month_sin"] = sin_term
    df["month_cos"] = cos_term
    return df


def create_lag_features(
    df: pd.DataFrame, target_col: str, group_col: str, lags: list
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        if lag < 1:
            raise ValueError("Lags must be >= 1 to avoid leaking the current target value.")
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str, group_col: str, windows: list
) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        # shift(1) ensures the current value never enters its own rolling mean
        df[f"{target_col}_rolling_mean_{window}"] = df.groupby(group_col)[target_col].transform(
            lambda x, w=window: x.shift(1).rolling(window=w).mean()
        )
    return df


def engineer_features(
    df: pd.DataFrame,
    cfg: DatasetConfig | None = None,
    target_col: str | None = None,
    lags: list | None = None,
    windows: list | None = None,
) -> pd.DataFrame:
    """Combine all feature engineering steps (calendar, seasonality, lags, rolling).

    Column names and window sizes default to the active dataset config;
    explicit arguments override it (useful for tests and experimentation).
    """
    cfg = cfg or get_config()
    target_col = target_col or cfg.target_col
    lags = lags or list(cfg.lags)
    windows = windows or list(cfg.rolling_windows)
    group_col = resolve_group_col(df, cfg)

    # Sort within each entity so shift/rolling operate on ordered history.
    df = df.sort_values(by=[group_col, cfg.date_col]).reset_index(drop=True)

    df = create_calendar_features(df, date_col=cfg.date_col)
    df = add_seasonality_terms(df)
    df = create_lag_features(df, target_col, group_col, lags)
    df = create_rolling_features(df, target_col, group_col, windows)

    # Drop warm-up rows created by shifts instead of imputing, which would leak.
    return df.dropna().reset_index(drop=True)

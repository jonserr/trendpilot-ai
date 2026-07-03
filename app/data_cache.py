"""Shared, cached data loading for all dashboard pages.

Every page imports from this module so the dataset is read and the features
are engineered exactly once per session, instead of once per page. Streamlit
shows the ``show_spinner`` message while each cache entry is being computed.
"""

import streamlit as st

from src.config import get_config
from src.data.features import engineer_features
from src.data.loaders import load_data


@st.cache_data(show_spinner="Loading dataset…")
def load_raw_dataset():
    """The validated raw dataset, or None when it hasn't been fetched yet."""
    cfg = get_config()
    if not cfg.data_path.exists():
        return None
    return load_data(cfg=cfg)


@st.cache_data(show_spinner="Engineering leakage-safe features…")
def load_engineered_dataset():
    """The engineered feature table (calendar, seasonality, lags, rolling)."""
    df = load_raw_dataset()
    if df is None:
        return None
    return engineer_features(df, cfg=get_config())


def missing_data_message(cfg=None) -> str:
    cfg = cfg or get_config()
    return (
        f"Dataset not found at `data/raw/{cfg.data_file}`. Run `make fetch-data` "
        "or place your own dataset there (see `config/dataset.yaml`)."
    )

from pathlib import Path

import pandas as pd

from src.config import DatasetConfig, get_config, validate_dataframe


def load_data(
    filepath: Path | None = None,
    cfg: DatasetConfig | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the configured CSV dataset, parsing dates and keeping entity
    identifiers as strings. Validates the schema against the dataset config."""
    cfg = cfg or get_config()
    filepath = Path(filepath) if filepath is not None else cfg.data_path
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    string_cols = {c: str for c in (*cfg.entity_hierarchy, *cfg.categorical_features)}
    df = pd.read_csv(filepath, dtype=string_cols)
    if validate:
        return validate_dataframe(df, cfg)
    if cfg.date_col in df.columns:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    return df


def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Saves a dataframe to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

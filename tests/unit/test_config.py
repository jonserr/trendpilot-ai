import pandas as pd
import pytest

from src.config import (
    DataValidationError,
    DatasetConfig,
    get_config,
    load_dataset_config,
    validate_dataframe,
)


def make_cfg(**overrides) -> DatasetConfig:
    defaults = dict(
        name="test",
        data_file="test.csv",
        date_col="date",
        target_col="value",
        entity_col="store",
        entity_hierarchy=("region", "store"),
        categorical_features=("region", "store"),
        numeric_features=("footfall",),
        lags=(1, 2),
        rolling_windows=(3,),
    )
    return DatasetConfig(**{**defaults, **overrides})


def make_df(**overrides) -> pd.DataFrame:
    data = {
        "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "region": ["north"] * 4,
        "store": ["s1"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0],
        "footfall": [10.0, 11.0, 12.0, 13.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_default_config_loads():
    cfg = load_dataset_config()
    assert cfg.name
    assert cfg.entity_col == cfg.entity_hierarchy[-1]
    assert cfg.target_col not in cfg.model_feature_cols


def test_get_config_is_cached():
    assert get_config() is get_config()


def test_derived_column_groups():
    cfg = make_cfg()
    assert cfg.lag_cols == ["value_lag_1", "value_lag_2"]
    assert cfg.rolling_cols == ["value_rolling_mean_3"]
    assert "month_sin" in cfg.model_feature_cols
    assert "month_sin" not in cfg.client_feature_cols
    assert "footfall" in cfg.client_feature_cols


def test_hierarchy_must_end_with_entity_col():
    with pytest.raises(DataValidationError):
        make_cfg(entity_hierarchy=("store", "region"))


def test_lags_must_be_positive():
    with pytest.raises(DataValidationError):
        make_cfg(lags=(0, 1))


def test_validate_dataframe_accepts_valid():
    df = validate_dataframe(make_df(), make_cfg())
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_validate_dataframe_missing_columns():
    df = make_df().drop(columns=["footfall"])
    with pytest.raises(DataValidationError, match="footfall"):
        validate_dataframe(df, make_cfg())


def test_validate_dataframe_non_numeric_target():
    df = make_df(value=["a", "b", "c", "d"])
    with pytest.raises(DataValidationError, match="numeric"):
        validate_dataframe(df, make_cfg())


def test_validate_dataframe_duplicate_rows():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_dataframe(df, make_cfg())

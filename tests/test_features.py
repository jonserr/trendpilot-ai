import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    create_calendar_features,
    create_lag_features,
    engineer_features,
)


def make_df(locations=("Loc One, AA",), periods=24, start="2022-01-01"):
    frames = []
    for i, loc in enumerate(locations):
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.date_range(start, periods=periods, freq="MS"),
                    "region": "AA",
                    "location": loc,
                    "median_sale_price": np.arange(periods, dtype=float) * 10 + 100 + i * 1000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_create_calendar_features():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"]})
    df_feat = create_calendar_features(df)

    assert df_feat["month"].tolist() == [1, 6]
    assert df_feat["quarter"].tolist() == [1, 2]
    assert df_feat["year"].tolist() == [2023, 2023]


def test_seasonality_terms_bounded_and_cyclical():
    df = engineer_features(make_df())
    assert {"month_sin", "month_cos"}.issubset(df.columns)
    assert df["month_sin"].between(-1, 1).all()
    assert df["month_cos"].between(-1, 1).all()
    # January maps to sin=0, cos=1
    january = df[df["month"] == 1].iloc[0]
    assert january["month_sin"] == pytest.approx(0.0, abs=1e-9)
    assert january["month_cos"] == pytest.approx(1.0, abs=1e-9)


def test_engineer_features_drops_warmup_rows():
    df = make_df(periods=10)
    df_feat = engineer_features(df)
    # The max window is 6 (with shift(1)), so the first 6 rows are dropped
    assert len(df_feat) == 4
    assert "median_sale_price_lag_1" in df_feat.columns
    assert "median_sale_price_rolling_mean_6" in df_feat.columns


def test_features_grouped_by_location_not_region():
    """Lags must never cross entity boundaries, even within the same region."""
    df = make_df(locations=("Loc One, AA", "Loc Two, AA"), periods=8)
    lagged = create_lag_features(
        df.sort_values(["location", "date"]), "median_sale_price", "location", [1]
    )
    first_rows = lagged.groupby("location").head(1)
    assert first_rows["median_sale_price_lag_1"].isna().all()


def test_no_future_leakage():
    """Features at time t must be identical whether or not future rows exist."""
    df_full = make_df(periods=24)
    df_trunc = df_full.iloc[:18].copy()

    feat_full = engineer_features(df_full)
    feat_trunc = engineer_features(df_trunc)

    overlap = len(feat_trunc)
    pd.testing.assert_frame_equal(
        feat_full.iloc[:overlap].reset_index(drop=True),
        feat_trunc.reset_index(drop=True),
    )


def test_no_leakage_when_future_values_change():
    """Perturbing future target values must not change past feature rows."""
    df = make_df(periods=24)
    df_perturbed = df.copy()
    df_perturbed.loc[df_perturbed.index[-1], "median_sale_price"] = 9_999_999.0

    feat_a = engineer_features(df)
    feat_b = engineer_features(df_perturbed)

    # All rows except the final one must be unaffected
    pd.testing.assert_frame_equal(feat_a.iloc[:-1], feat_b.iloc[:-1])


def test_lag_zero_rejected():
    df = make_df()
    with pytest.raises(ValueError):
        create_lag_features(df, "median_sale_price", "location", [0])

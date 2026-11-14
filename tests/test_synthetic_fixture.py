"""Tests for the offline test-fixture generator (tests/fixtures/synthetic.py).

The fixture mirrors the default dataset schema so the pipeline can be
exercised without network access; these tests keep it honest.
"""

import pandas as pd

from src.config import get_config, validate_dataframe
from tests.fixtures.synthetic import COLUMNS, generate_data


def small_df(**kwargs):
    geography = {
        "AA": ["City One, AA", "City Two, AA"],
        "BB": ["City Three, BB"],
    }
    return generate_data(years=2, seed=7, geography=geography, **kwargs)


def test_expected_columns_present():
    df = small_df()
    assert list(df.columns) == COLUMNS


def test_fixture_matches_dataset_config_schema():
    df = small_df()
    validate_dataframe(df, get_config())  # raises on mismatch


def test_multiple_locations_per_region():
    df = small_df()
    locations_per_region = df.groupby("region")["location"].nunique()
    assert locations_per_region["AA"] == 2
    assert pd.api.types.is_string_dtype(df["location"])
    assert df["location"].map(lambda value: isinstance(value, str)).all()


def test_monthly_continuity_per_location():
    df = small_df()
    for _, group in df.groupby("location"):
        dates = pd.DatetimeIndex(sorted(group["date"]))
        assert len(dates) == 24  # 2 years of monthly data
        expected = pd.date_range(dates[0], periods=24, freq="MS")
        assert dates.equals(expected)


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(small_df(), small_df())


def test_value_ranges():
    df = small_df()
    assert (df["median_sale_price"] >= 75_000).all()
    assert df["mortgage_rate"].between(2.5, 8.0).all()
    assert df["sale_to_list_ratio"].between(0.85, 1.10).all()
    assert (df["inventory"] > 0).all()
    assert (df["new_listings"] > 0).all()

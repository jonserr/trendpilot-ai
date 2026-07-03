"""Offline unit tests for the real-data ingestion pipeline (no network)."""

import pandas as pd
import pytest

from src.data.ingest import (
    build_dataset,
    fetch_mortgage_rates,
    load_redfin,
    normalize_redfin_chunk,
)


def make_redfin_chunk(
    n_months=4, region="Seattle, WA metro area", state="WA", price="800000"
) -> pd.DataFrame:
    periods = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    return pd.DataFrame(
        {
            "period_begin": periods - pd.DateOffset(months=2),
            "period_end": periods + pd.offsets.MonthEnd(0),
            "region": [region] * n_months,
            "state_code": [state] * n_months,
            "property_type": ["All Residential"] * n_months,
            "is_seasonally_adjusted": ["f"] * n_months,
            "median_sale_price": [price] * n_months,
            "inventory": ["120"] * n_months,
            "median_dom": ["12"] * n_months,
            "avg_sale_to_list": ["1.02"] * n_months,
            "new_listings": ["45"] * n_months,
        }
    )


def test_normalize_extracts_canonical_columns():
    out = normalize_redfin_chunk(make_redfin_chunk())
    assert set(out.columns) == {
        "date",
        "region",
        "location",
        "median_sale_price",
        "inventory",
        "days_on_market",
        "sale_to_list_ratio",
        "new_listings",
    }
    assert (out["location"] == "Seattle, WA").all()  # " metro area" suffix stripped
    assert (out["region"] == "WA").all()
    assert out["median_sale_price"].dtype.kind in "if"


def test_normalize_zip_level_extracts_zipcode():
    chunk = make_redfin_chunk(region="Zip Code: 98101")
    out = normalize_redfin_chunk(chunk, level="zip")
    assert (out["location"] == "98101").all()
    assert (out["region"] == "WA").all()


def test_normalize_parses_state_from_region_when_state_code_missing():
    chunk = make_redfin_chunk().drop(columns=["state_code"])
    out = normalize_redfin_chunk(chunk)
    assert (out["region"] == "WA").all()


def test_normalize_filters_property_type_and_adjustment():
    chunk = make_redfin_chunk()
    chunk.loc[0, "property_type"] = "Condo/Co-op"
    chunk.loc[1, "is_seasonally_adjusted"] = "t"
    out = normalize_redfin_chunk(chunk)
    assert len(out) == 2


def test_normalize_filters_states_and_locations():
    chunk = pd.concat(
        [
            make_redfin_chunk(region="Seattle, WA metro area", state="WA"),
            make_redfin_chunk(region="Portland, OR metro area", state="OR"),
        ],
        ignore_index=True,
    )
    assert normalize_redfin_chunk(chunk, states=["OR"])["location"].unique().tolist() == [
        "Portland, OR"
    ]
    assert normalize_redfin_chunk(chunk, locations=["seattle, wa"])[
        "location"
    ].unique().tolist() == ["Seattle, WA"]


def test_normalize_handles_uppercase_headers():
    """Redfin has changed header casing before; matching must be case-insensitive."""
    chunk = make_redfin_chunk()
    chunk.columns = [c.upper() for c in chunk.columns]
    out = normalize_redfin_chunk(chunk)
    assert len(out) == 4
    assert (out["location"] == "Seattle, WA").all()


def test_fetch_mortgage_rates_resamples_monthly(tmp_path):
    csv = tmp_path / "rates.csv"
    csv.write_text(
        "observation_date,MORTGAGE30US\n"
        "2024-01-04,6.62\n2024-01-11,6.66\n2024-01-18,6.60\n2024-02-01,6.63\n"
    )
    rates = fetch_mortgage_rates(str(csv))
    assert list(rates.columns) == ["date", "mortgage_rate"]
    assert len(rates) == 2
    jan = rates.loc[rates["date"] == "2024-01-01", "mortgage_rate"].iloc[0]
    assert jan == pytest.approx(6.63, abs=0.01)


def _write_redfin_file(path, uppercase=True, drop_col=None, quote_all=False):
    import csv
    import gzip as gz

    df = pd.concat(
        [
            make_redfin_chunk(n_months=4, region="Seattle, WA metro area"),
            make_redfin_chunk(n_months=4, region="Portland, OR metro area", state="OR"),
        ],
        ignore_index=True,
    )
    if drop_col:
        df = df.drop(columns=[drop_col])
    if uppercase:
        df.columns = [c.upper() for c in df.columns]
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    content = df.to_csv(sep="\t", index=False, quoting=quoting)
    if str(path).endswith(".gz"):
        with gz.open(path, "wt") as f:
            f.write(content)
    else:
        path.write_text(content)
    return path


def test_load_redfin_local_gz_with_uppercase_headers(tmp_path):
    source = _write_redfin_file(tmp_path / "tracker.tsv.gz", uppercase=True)
    df = load_redfin(source=str(source))
    assert sorted(df["location"].unique()) == ["Portland, OR", "Seattle, WA"]
    assert len(df) == 8  # one row per location-month after dedupe


def test_load_redfin_handles_quoted_uppercase_headers(tmp_path):
    """Regression: the live export quotes every field ('"REGION"'); pandas
    strips quotes at parse time, so column selection must canonicalize."""
    source = _write_redfin_file(tmp_path / "tracker.tsv.gz", uppercase=True, quote_all=True)
    df = load_redfin(source=str(source))
    assert sorted(df["location"].unique()) == ["Portland, OR", "Seattle, WA"]
    assert df["median_sale_price"].gt(0).all()


def test_load_redfin_reports_missing_columns(tmp_path):
    source = _write_redfin_file(tmp_path / "tracker.tsv", uppercase=True, drop_col="median_dom")
    with pytest.raises(SystemExit, match="median_dom"):
        load_redfin(source=str(source))


def test_load_redfin_rejects_unknown_level():
    with pytest.raises(SystemExit, match="level"):
        load_redfin(level="galaxy")


def test_build_dataset_merges_and_filters_sparse_series():
    redfin = pd.concat(
        [
            normalize_redfin_chunk(make_redfin_chunk(n_months=6)),
            normalize_redfin_chunk(
                make_redfin_chunk(n_months=2, region="Portland, OR metro area", state="OR")
            ),
        ],
        ignore_index=True,
    )
    rates = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "mortgage_rate": [6.6, 6.7, 6.8, 6.9, 7.0, 7.1],
        }
    )
    df = build_dataset(redfin, rates, min_months=3, max_entities=None)

    assert df["location"].unique().tolist() == ["Seattle, WA"]  # sparse series dropped
    assert "mortgage_rate" in df.columns
    assert list(df.columns) == [
        "date",
        "region",
        "location",
        "median_sale_price",
        "inventory",
        "mortgage_rate",
        "days_on_market",
        "sale_to_list_ratio",
        "new_listings",
    ]


def test_build_dataset_raises_when_everything_filtered():
    redfin = normalize_redfin_chunk(make_redfin_chunk(n_months=2))
    rates = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=2, freq="MS"), "mortgage_rate": [6.6, 6.7]}
    )
    with pytest.raises(SystemExit):
        build_dataset(redfin, rates, min_months=12)

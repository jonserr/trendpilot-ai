"""Deterministic housing-panel fixture generator for tests.

Produces data matching the default dataset schema (see config/dataset.yaml)
so the test suite can exercise the full pipeline offline — no network access
or real dataset required. Not part of the product: real data comes from
``src/data/ingest.py`` or a user-provided CSV.
"""

import numpy as np
import pandas as pd

# region (state) -> locations (metros)
GEOGRAPHY = {
    "WA": ["Seattle, WA", "Tacoma, WA"],
    "OR": ["Portland, OR", "Salem, OR"],
    "CO": ["Denver, CO", "Boulder, CO"],
    "ID": ["Boise, ID", "Idaho Falls, ID"],
}

COLUMNS = [
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


def generate_data(
    start_date: str = "2019-01-01",
    years: int = 6,
    seed: int = 42,
    geography: dict | None = None,
) -> pd.DataFrame:
    """Generate a monthly, location-level housing panel with per-location
    price levels, trends, seasonality, rate sensitivity, and seeded noise."""
    rng = np.random.default_rng(seed)
    geography = geography or GEOGRAPHY

    periods = years * 12
    dates = pd.date_range(start=pd.to_datetime(start_date), periods=periods, freq="MS")

    # National mortgage-rate path shared by all locations.
    rate_path = 3.5 + np.sin(np.linspace(0, np.pi, periods)) * 1.5
    rate_path = rate_path + np.cumsum(rng.normal(0, 0.08, periods))
    rate_path = np.clip(rate_path, 2.5, 8.0)

    records = []
    for region, locations in geography.items():
        region_multiplier = rng.uniform(0.8, 1.3)
        region_base_inventory = rng.uniform(400, 4_000)
        region_base_dom = rng.uniform(15, 60)
        for location in locations:
            premium = rng.uniform(0.85, 1.25)
            base_price = rng.uniform(250_000, 750_000) * region_multiplier * premium
            total_appreciation = rng.uniform(0.15, 0.45)
            trend = np.linspace(0, base_price * total_appreciation, periods)
            phase = rng.uniform(-0.5, 0.5)
            seasonality = (
                np.sin(2 * np.pi * (np.arange(periods) / 12.0 + phase)) * base_price * 0.04
            )
            noise = rng.normal(0, base_price * 0.015, periods)

            for i, date in enumerate(dates):
                rate_impact = -(rate_path[i] - 4.0) * 0.015 * base_price
                price = base_price + trend[i] + seasonality[i] + noise[i] + rate_impact
                price = max(75_000.0, price)

                seasonal_ratio = seasonality[i] / base_price
                inventory = int(
                    max(
                        25,
                        region_base_inventory * premium * rng.uniform(0.9, 1.1)
                        + rate_path[i] * 40
                        - seasonal_ratio * 200,
                    )
                )
                days_on_market = int(
                    max(
                        5,
                        region_base_dom
                        + rate_path[i] * 2
                        - seasonal_ratio * 150
                        + rng.normal(0, 4),
                    )
                )
                sale_to_list_ratio = float(
                    np.clip(0.97 + seasonal_ratio + rng.normal(0, 0.015), 0.85, 1.10)
                )
                new_listings = int(max(5, inventory * rng.uniform(0.12, 0.28)))

                records.append(
                    {
                        "date": date,
                        "region": region,
                        "location": location,
                        "median_sale_price": round(price, 2),
                        "inventory": inventory,
                        "mortgage_rate": round(float(rate_path[i]), 2),
                        "days_on_market": days_on_market,
                        "sale_to_list_ratio": round(sale_to_list_ratio, 3),
                        "new_listings": new_listings,
                    }
                )

    df = pd.DataFrame(records, columns=COLUMNS)
    return df.sort_values(["location", "date"]).reset_index(drop=True)

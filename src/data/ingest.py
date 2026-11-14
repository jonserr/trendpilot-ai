"""Real-data ingestion for the default housing dataset.

Downloads housing market data published by Redfin
(https://www.redfin.com/news/data-center/) and 30-year fixed mortgage rates
published by Freddie Mac via FRED (series MORTGAGE30US), then normalizes both
into the canonical schema described in ``config/dataset.yaml``:

    date, region, location, median_sale_price, inventory, mortgage_rate,
    days_on_market, sale_to_list_ratio, new_listings

Redfin publishes the same metrics at several geographic levels. The default
is ``metro`` (~110 MB download, a few hundred metro areas) which is plenty
for demos and experimentation; ``--level state`` (~5 MB) is even smaller,
and ``zip`` (~1 GB), ``county``, and ``city`` are also supported. All levels
produce the same schema, so no config change is needed. Both sources update monthly, require no API key, and are free for
non-commercial use with attribution; cite Redfin when publishing results.

Downloads are resumable and cached under ``data/raw/.cache/``; pass
``--refresh`` to force a fresh copy. Column matching is case-insensitive so
header-casing changes in the export do not break parsing.

Usage (inside the API container via ``make fetch-data`` or directly):

    python src/data/ingest.py --states WA OR                 # metro level
    python src/data/ingest.py --level zip --states WA        # ZIP level
    python src/data/ingest.py --locations "Seattle, WA" "Portland, OR"

To forecast a different dataset entirely, skip this module: place your own
CSV at the path configured in ``config/dataset.yaml`` (or point
``TRENDPILOT_CONFIG`` at your own config).
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import get_config, validate_dataframe  # noqa: E402
from src.data.loaders import save_data  # noqa: E402
from src.utils.paths import RAW_DATA_DIR, SAMPLE_DATA_DIR, ensure_dirs  # noqa: E402

_REDFIN_BASE = "https://redfin-public-data.s3-us-west-2.amazonaws.com/redfin_market_tracker"
REDFIN_URLS = {
    "metro": f"{_REDFIN_BASE}/redfin_metro_market_tracker.tsv000.gz",  # ~110 MB
    "state": f"{_REDFIN_BASE}/state_market_tracker.tsv000.gz",  # ~5 MB
    "county": f"{_REDFIN_BASE}/county_market_tracker.tsv000.gz",  # ~200 MB
    "city": f"{_REDFIN_BASE}/city_market_tracker.tsv000.gz",  # ~600 MB
    "zip": f"{_REDFIN_BASE}/zip_code_market_tracker.tsv000.gz",  # ~1 GB
}
DEFAULT_LEVEL = "metro"

FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

# Redfin metric column -> canonical column (matched case-insensitively)
REDFIN_COLUMN_MAP = {
    "median_sale_price": "median_sale_price",
    "inventory": "inventory",
    "median_dom": "days_on_market",
    "avg_sale_to_list": "sale_to_list_ratio",
    "new_listings": "new_listings",
}
REDFIN_USECOLS = [
    "period_begin",
    "period_end",
    "region",
    "state_code",
    "property_type",
    "is_seasonally_adjusted",
    *REDFIN_COLUMN_MAP.keys(),
]
# Columns that must exist (case-insensitively) for ingestion to proceed.
REQUIRED_REDFIN_COLS = ["region", *REDFIN_COLUMN_MAP.keys()]

CHUNK_SIZE = 250_000


def _canonical(name: str) -> str:
    """Normalize a header name: strip whitespace/quotes/BOM, lowercase."""
    return str(name).strip().strip('"').lstrip("﻿").strip().lower()


def _cache_path(level: str) -> Path:
    return RAW_DATA_DIR / ".cache" / f"redfin_{level}_tracker.tsv.gz"


def fetch_mortgage_rates(source: str = FRED_MORTGAGE_URL) -> pd.DataFrame:
    """Monthly average 30-year fixed mortgage rate from FRED (no API key).

    ``source`` may be a URL or a local CSV path with (date, rate) columns.
    """
    raw = pd.read_csv(source)
    raw.columns = ["date", "mortgage_rate"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["mortgage_rate"] = pd.to_numeric(raw["mortgage_rate"], errors="coerce")
    raw = raw.dropna()

    monthly = raw.set_index("date").resample("MS")["mortgage_rate"].mean().round(2).reset_index()
    return monthly


def _extract_location(region: pd.Series, level: str) -> pd.Series:
    """Entity identifier from Redfin's region label for the given level."""
    s = region.astype(str).str.strip()
    if level == "zip":
        return s.str.extract(r"(\d{5})")[0]
    # e.g. "Seattle, WA metro area" -> "Seattle, WA"
    return s.str.replace(r"\s+metro area$", "", regex=True).str.strip()


def _extract_region(df: pd.DataFrame, level: str) -> pd.Series:
    """Coarse grouping (state code) for the hierarchy's first level."""
    if "state_code" in df.columns:
        state = df["state_code"].astype(str).str.strip()
        if not state.replace("", pd.NA).isna().all():
            return state
    if level == "state":
        return df["region"].astype(str).str.strip()
    # Fall back to the ", XX" suffix in the region label.
    return df["region"].astype(str).str.extract(r",\s*([A-Z]{2})\b")[0]


def normalize_redfin_chunk(
    chunk: pd.DataFrame,
    level: str = DEFAULT_LEVEL,
    states: list | None = None,
    locations: list | None = None,
    start: str | None = None,
) -> pd.DataFrame:
    """Normalize one chunk of a Redfin market tracker into the canonical
    schema (without the mortgage rate, merged later)."""
    df = chunk.rename(columns={c: _canonical(c) for c in chunk.columns})
    if "property_type" in df.columns:
        df = df[df["property_type"] == "All Residential"]
    if "is_seasonally_adjusted" in df.columns:
        df = df[df["is_seasonally_adjusted"].astype(str).str.lower().isin(["f", "false"])]

    df = df.copy()
    df["location"] = _extract_location(df["region"], level)
    df["region_code"] = _extract_region(df, level)
    df = df.dropna(subset=["location", "region_code"])

    if states:
        df = df[df["region_code"].isin(states)]
    if locations:
        wanted = {loc.casefold() for loc in locations}
        df = df[df["location"].str.casefold().isin(wanted)]
    if df.empty:
        return pd.DataFrame()

    # Assign each rolling window to the month it ends in.
    date_col = "period_end" if "period_end" in df.columns else "period_begin"
    df["date"] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]

    out = pd.DataFrame(
        {
            "date": df["date"],
            "region": df["region_code"].astype(str),
            "location": df["location"].astype(str),
        }
    )
    for src, dst in REDFIN_COLUMN_MAP.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    return out.dropna()


def _download(url: str, dest: Path, attempts: int = 6) -> Path:
    """Resumable download: retries with HTTP Range requests on read timeouts
    or dropped connections, so an interrupted transfer picks up where it left
    off instead of starting over."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        resume_at = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        try:
            with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as response:
                if response.status_code == 416:
                    # Requested range beyond EOF: the file is already complete.
                    print("  download already complete (cached).")
                    return dest
                response.raise_for_status()

                # Server ignored the Range header -> restart from scratch.
                mode = "ab" if resume_at and response.status_code == 206 else "wb"
                done = resume_at if mode == "ab" else 0
                total = int(response.headers.get("Content-Length", 0)) + done

                with open(dest, mode) as f:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        f.write(block)
                        done += len(block)
                        progress = f" of {total / 1e6:,.0f}" if total else ""
                        print(f"  {done / 1e6:,.0f}{progress} MB", end="\r")
            print()
            return dest
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                print(f"\n  transfer interrupted ({exc}); resuming ({attempt}/{attempts - 1})...")
    raise SystemExit(
        f"Download failed after {attempts} attempts: {last_error}\n"
        f"Partial file kept at {dest} — re-running will resume from there."
    )


def _resolve_source(source: str, level: str, refresh: bool = False) -> Path:
    """Return a local path for ``source``, downloading (with cache/resume)
    when it is a URL."""
    if not str(source).startswith(("http://", "https://")):
        return Path(source)

    cache = _cache_path(level)
    if refresh and cache.exists():
        cache.unlink()
    if cache.exists():
        print(f"Using cached download {cache} (pass --refresh to re-download).")
        return cache
    return _download(source, cache)


def _inspect_header(path: Path) -> tuple[str, list]:
    """Sniff the delimiter and validate that the required columns exist
    (case-insensitively). Returns (sep, usecols original names)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        first_line = f.readline()

    sep = "\t" if first_line.count("\t") >= first_line.count(",") else ","
    header = [h for h in first_line.rstrip("\r\n").split(sep)]
    canon_to_original = {_canonical(h): h for h in header}

    missing = [c for c in REQUIRED_REDFIN_COLS if c not in canon_to_original]
    if "period_begin" not in canon_to_original and "period_end" not in canon_to_original:
        missing.append("period_begin/period_end")
    if missing:
        raise SystemExit(
            f"Redfin file at {path} is missing expected columns: {missing}\n"
            f"Columns found: {sorted(canon_to_original)}\n"
            "Redfin may have changed the export format — update REDFIN_COLUMN_MAP / "
            "REDFIN_USECOLS in src/data/ingest.py accordingly."
        )

    return sep


def load_redfin(
    source: str | None = None,
    level: str = DEFAULT_LEVEL,
    states: list | None = None,
    locations: list | None = None,
    start: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Stream a Redfin market tracker file and return normalized rows.

    ``source`` may be a URL or a local .tsv/.tsv.gz path; defaults to the
    public URL for ``level``.
    """
    if level not in REDFIN_URLS:
        raise SystemExit(f"Unknown --level '{level}'. Choose from: {sorted(REDFIN_URLS)}")
    source = source or REDFIN_URLS[level]
    path = _resolve_source(source, level, refresh=refresh)
    sep = _inspect_header(path)

    opener = gzip.open if str(path).endswith(".gz") else open
    frames = []
    with opener(path, "rt", errors="replace") as f:
        # Select columns with a canonicalizing callable: pandas strips CSV
        # quotes from header names during parsing, so literal names from the
        # raw header line (e.g. '"REGION"') would not match.
        reader = pd.read_csv(
            f,
            sep=sep,
            chunksize=CHUNK_SIZE,
            usecols=lambda c: _canonical(c) in REDFIN_USECOLS,
            dtype=str,
        )
        for i, chunk in enumerate(reader):
            normalized = normalize_redfin_chunk(
                chunk, level=level, states=states, locations=locations, start=start
            )
            if not normalized.empty:
                frames.append(normalized)
            print(f"  processed chunk {i + 1} ({len(frames)} with matching rows)", end="\r")
    print()
    if not frames:
        raise SystemExit(
            "No rows matched the requested filters. Check --states / --locations values."
        )

    df = pd.concat(frames, ignore_index=True)
    # Redfin publishes overlapping rolling windows; keep one row per
    # (location, month) by averaging duplicates.
    df = (
        df.groupby(["location", "date"], as_index=False)
        .agg(
            {
                "region": "first",
                "median_sale_price": "mean",
                "inventory": "mean",
                "days_on_market": "mean",
                "sale_to_list_ratio": "mean",
                "new_listings": "mean",
            }
        )
        .round(
            {
                "median_sale_price": 2,
                "inventory": 0,
                "days_on_market": 0,
                "sale_to_list_ratio": 4,
                "new_listings": 0,
            }
        )
    )
    return df


def build_dataset(
    redfin_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    min_months: int = 36,
    max_entities: int | None = None,
) -> pd.DataFrame:
    """Merge sources, drop sparse series, and order canonical columns."""
    df = redfin_df.merge(rates_df, on="date", how="inner")

    # Keep only series with enough history to engineer lag/rolling features.
    counts = df.groupby("location")["date"].count()
    keep = counts[counts >= min_months]
    if max_entities:
        keep = keep.sort_values(ascending=False).head(max_entities)
    df = df[df["location"].isin(keep.index)]
    if df.empty:
        raise SystemExit(
            f"No locations have at least {min_months} months of data after filtering. "
            "Loosen --min-months or widen the geographic filters."
        )

    columns = [
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
    return df[columns].sort_values(["location", "date"]).reset_index(drop=True)


def run_ingestion(args) -> pd.DataFrame:
    cfg = get_config()
    ensure_dirs()

    rates = fetch_mortgage_rates(args.rates_source)
    redfin = load_redfin(
        source=args.source,
        level=args.level,
        states=args.states,
        locations=args.locations,
        start=args.start,
        refresh=args.refresh,
    )
    df = build_dataset(redfin, rates, min_months=args.min_months, max_entities=args.max_entities)

    df = validate_dataframe(df, cfg)
    save_data(df, cfg.data_path)
    print(
        f"Saved {len(df):,} rows / {df['location'].nunique()} locations "
        f"({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}) -> {cfg.data_path}"
    )

    # Refresh the small sample used for schema reference.
    sample_locations = df["location"].drop_duplicates().head(2)
    sample = df[df["location"].isin(sample_locations)]
    save_data(sample, SAMPLE_DATA_DIR / "sample_housing_data.csv")
    return df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        choices=sorted(REDFIN_URLS),
        help=f"Redfin geographic level (default: {DEFAULT_LEVEL}, ~110 MB download)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override: Redfin tracker URL or local .tsv/.tsv.gz path",
    )
    parser.add_argument(
        "--rates-source",
        default=FRED_MORTGAGE_URL,
        help="FRED MORTGAGE30US CSV URL or local CSV path",
    )
    parser.add_argument("--states", nargs="*", default=None, help="State codes, e.g. WA OR")
    parser.add_argument(
        "--locations",
        nargs="*",
        default=None,
        help='Exact locations to keep, e.g. "Seattle, WA" (metro) or 98101 (zip)',
    )
    parser.add_argument("--start", default="2016-01-01", help="Earliest observation month")
    parser.add_argument(
        "--min-months", type=int, default=36, help="Minimum months of history per location"
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=50,
        help="Keep at most N locations (densest history first); 0 = no limit",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Discard the cached Redfin download and fetch a fresh copy",
    )
    args = parser.parse_args(argv)
    if args.max_entities == 0:
        args.max_entities = None
    return args


if __name__ == "__main__":
    from src.modeling.artifacts import TrainingInProgressError, training_lock

    args = parse_args()
    try:
        with training_lock():
            run_ingestion(args)
    except TrainingInProgressError as exc:
        raise SystemExit(str(exc)) from None

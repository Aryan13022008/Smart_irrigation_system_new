"""
Stage 02 — cleaning, feature engineering and targets.

Takes the merged per-sector tables and produces the analysis-ready tables: soil
moisture is cleaned of sensor faults, short gaps are interpolated, calendar
features are derived, and each row is given its 24-hour-ahead targets together
with the water volume applied over the horizon.

Two properties are worth stating explicitly, because they are easy to get wrong.

Cleaning removes rows rather than imputing them. Soil moisture is the quantity
being modelled, so inventing values for it would leak into both the features and
the targets; a shorter, honest series is preferable.

Forward and backward windows are therefore computed on the regular 10-minute
grid, not on the cleaned series. Counting 144 rows of a series with holes in it
would span more than 24 hours; counting 144 samples of the grid always spans
exactly 24 hours. Rows whose window is not fully covered are dropped rather than
averaged over whatever happens to be present.

Inputs
    data/merged/dataset_zone_{1..5}.csv

Outputs
    data/preprocessed/dataset_zone_{1..5}_preprocessed.csv

Run from anywhere:  python 02_preprocessing.py
"""

import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# Loading and calendar features
# ---------------------------------------------------------------------------
def load_grid(sector):
    """Load one merged table onto a gap-free 10-minute grid.

    Missing sensor readings stay missing; only irrigation is known to be zero
    where nothing was recorded, because the signal was reconstructed from an
    explicit event list rather than sampled.
    """
    path = cfg.MERGED_DIR / f"dataset_zone_{sector}.csv"
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"{path.name} has no 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")

    grid = pd.date_range(df.index.min(), df.index.max(), freq=cfg.SAMPLING_INTERVAL)
    df = df.reindex(grid)
    df.index.name = "ts"
    df["irrigation_duration_minutes"] = df["irrigation_duration_minutes"].fillna(0)
    return df


def add_calendar_features(df):
    """Season and time-of-day descriptors."""
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["day_period"] = df["hour"].map(cfg.day_period)
    df["quarter"] = df.index.quarter
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def interpolate_short_gaps(df):
    """Bridge gaps of at most INTERPOLATION_LIMIT samples in the soil sensors."""
    columns = [c for c in cfg.INTERPOLATE_COLUMNS if c in df.columns]
    if columns:
        df[columns] = df[columns].interpolate(method="time", limit=cfg.INTERPOLATION_LIMIT)
    return df


def valid_soil_sensors(df):
    """Rows where every soil sensor reports a plausible value.

    Out-of-band pH and EC readings are error codes, not measurements: the probes
    return values consistent with a 16-bit overflow when they lose contact.
    """
    mask = df[list(cfg.INTERPOLATE_COLUMNS)].notna().all(axis=1)
    for column, (low, high) in cfg.SENSOR_VALID_RANGES.items():
        mask &= df[column].between(low, high)
    low, high = cfg.MOISTURE_VALID_RANGE
    return mask & df["soil_moisture"].between(low, high)


def cap_saturation(df):
    """Clip the over-saturation artefact of the moisture probe."""
    df.loc[df["soil_moisture"] > cfg.MOISTURE_CLIP_MAX, "soil_moisture"] = cfg.MOISTURE_CLIP_MAX
    return df


def remove_spikes(df, column="soil_moisture"):
    """Drop readings that jump away from their neighbourhood and back.

    A probe losing contact with the soil produces a single sample far from the
    local level; a real wetting front does not.
    """
    median = df[column].rolling(window=cfg.SPIKE_WINDOW, center=True).median()
    return df[~((df[column] - median).abs() > cfg.SPIKE_THRESHOLD)]


def trim_trailing_flatline(df, column="soil_moisture"):
    """Cut the tail of the series once the probe stops varying at all, which
    means it has stalled rather than that the soil has stabilised."""
    reversed_std = df[column].iloc[::-1].rolling(window=cfg.FLATLINE_WINDOW).std()
    varying = reversed_std[reversed_std > 0.001]
    if varying.empty:
        return df
    return df[df.index <= varying.index[0]]


# ---------------------------------------------------------------------------
# Targets and water volumes
# ---------------------------------------------------------------------------
def add_targets_and_water(df, steps, past_steps, flow_rate):
    """Add the forward-looking targets and the water volume features.

    ``df`` must be indexed on the regular grid, so that a window of ``steps``
    rows is also a window of ``steps`` sampling intervals.
    """
    horizon = cfg.TARGET_HORIZON_HOURS
    forward = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)
    min_samples = int(steps * cfg.MIN_WINDOW_COVERAGE)

    df[f"target_point_{horizon}h"] = df["soil_moisture"].shift(-steps)
    df[f"target_mean_{horizon}h"] = df["soil_moisture"].rolling(
        window=forward, min_periods=min_samples
    ).mean()

    # Irrigation is known everywhere on the grid, so its windows are always full.
    litres = df["irrigation_duration_minutes"] * flow_rate
    df[f"water_vol_to_{horizon}h"] = litres.rolling(window=forward).sum()
    df[f"water_vol_past_{cfg.PAST_WATER_WINDOW_HOURS}h"] = litres.rolling(
        window=past_steps, min_periods=1
    ).sum()

    # Realised change over the horizon. Diagnostic only: it is the outcome the
    # models are meant to predict, so it must never be used as an input feature.
    df["real_moisture_delta"] = df["soil_moisture"].shift(-steps) - df["soil_moisture"]
    return df


def report_consistency(df):
    """Sanity check: applying more water should raise soil moisture."""
    horizon = cfg.TARGET_HORIZON_HOURS
    pair = df[[f"water_vol_to_{horizon}h", "real_moisture_delta"]].dropna()
    if len(pair) < 10:
        return
    correlation = pair.corr(method="spearman").iloc[0, 1]
    flag = "" if correlation > 0 else "   <-- unexpected sign, inspect this sector"
    print(f"    Water vs. moisture change, Spearman rho = {correlation:+.3f}{flag}")


# ---------------------------------------------------------------------------
# Per-sector pipeline
# ---------------------------------------------------------------------------
def process_sector(sector):
    horizon = cfg.TARGET_HORIZON_HOURS
    steps = horizon * 60 // cfg.SAMPLING_RATE_MINUTES
    past_steps = cfg.PAST_WATER_WINDOW_HOURS * 60 // cfg.SAMPLING_RATE_MINUTES

    print(f"--- Sector {sector} ---")
    grid = load_grid(sector)
    print(f"    {len(grid)} grid rows from {grid.index.min()} to {grid.index.max()}")

    grid = add_calendar_features(grid)
    grid = interpolate_short_gaps(grid)
    grid = grid.drop(columns=list(cfg.COLS_TO_DROP), errors="ignore")
    grid = cap_saturation(grid)

    # Which rows survive cleaning. The spike and flatline filters are defined on
    # the cleaned series, so they are applied to it and the outcome is mapped
    # back onto the grid.
    cleaned = grid[valid_soil_sensors(grid)]
    cleaned = remove_spikes(cleaned)
    cleaned = trim_trailing_flatline(cleaned)
    print(f"    {len(cleaned)} rows after cleaning")

    # Discarded rows must not contribute to any window, so their soil moisture is
    # blanked on the grid before the windows are taken.
    grid.loc[~grid.index.isin(cleaned.index), "soil_moisture"] = pd.NA
    grid["soil_moisture"] = grid["soil_moisture"].astype(float)
    grid = add_targets_and_water(grid, steps, past_steps, cfg.flow_rate(sector))

    df = grid.loc[cleaned.index]
    df = df.dropna(subset=[f"target_point_{horizon}h", f"target_mean_{horizon}h"])
    print(f"    {len(df)} rows with a usable {horizon} h horizon")

    if df.empty:
        # The sector was sampled too sparsely for any window to qualify. Writing
        # an empty table would suggest the sector is usable, so it is skipped.
        print(f"    No usable rows — sector {sector} is not written.")
        return df

    report_consistency(df)
    path = cfg.PREPROCESSED_DIR / f"dataset_zone_{sector}_preprocessed.csv"
    df.to_csv(path)
    print(f"    Saved {path.relative_to(cfg.ROOT)}")
    return df


def main():
    cfg.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for sector in cfg.SECTORS:
        path = cfg.MERGED_DIR / f"dataset_zone_{sector}.csv"
        if not path.exists():
            print(f"--- Sector {sector} --- missing {path.name}, skipping.")
            continue
        df = process_sector(sector)
        if df.empty:
            continue
        summary.append({
            "sector": sector,
            "rows": len(df),
            "from": df.index.min(),
            "to": df.index.max(),
            "soil_type": cfg.SECTOR_SOIL_TYPE[sector],
        })

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print("\nStage 02 complete.")


if __name__ == "__main__":
    main()

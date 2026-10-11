"""
Stage 01 — merging.

Combines the raw sensor exports, the validated irrigation events from stage 00
and reanalysis weather data into one regularly sampled table per sector.

Sensors report irregularly and independently, so every reading is rounded to the
minute, averaged per node, and then aggregated onto a common 10-minute grid.
Irrigation is not a sensor reading: it is reconstructed by expanding each
validated event into a minute-level valve signal and counting, for every bin,
how many minutes the valve was open.

Requires an internet connection: weather variables are fetched from the
Open-Meteo historical archive. See the note on reproducibility in
03_code/README.md.

Inputs
    data/original/raw_sensor/MEASUREMENT_{ste,sti,sue,sui,ce,ph,sut}.csv
    data/irrigation_actuators/final_irrigation_events.csv

Outputs
    data/merged/dataset_zone_{1..5}.csv

Run from anywhere:  python 01_merge_datasets.py
"""

import pandas as pd
import requests

import config as cfg


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
def load_sensor(tag, keep_node=False):
    """Load one raw sensor export and reduce it to one value per minute per node.

    ``keep_node`` retains the node identifier, which is needed for the sensors
    that exist once per sector.
    """
    column = cfg.SENSOR_COLUMNS[tag]
    path = cfg.RAW_SENSOR_DIR / cfg.SENSOR_FILES[tag]
    if not path.exists():
        print(f"  [WARN] Missing {path.name}, skipping {column}.")
        return None

    df = pd.read_csv(path, low_memory=False, dtype={"fk_iot_node": str})
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"time": "ts", "measurement_date": "ts"})

    # Soil moisture is reported in its own column; every other sensor uses `value`.
    source = "humidity_percentage" if column == "soil_moisture" else "value"
    df = df.rename(columns={source: column})

    df = df[[c for c in ("ts", "fk_iot_node", column) if c in df.columns]]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["ts"] = df["ts"].dt.round("min")

    group_keys = ["ts", "fk_iot_node"] if keep_node and "fk_iot_node" in df.columns else ["ts"]
    df = df.groupby(group_keys, as_index=False).mean(numeric_only=True)

    if not keep_node:
        df = df.drop(columns=["fk_iot_node"], errors="ignore")
    return df


def load_farm_wide_sensors():
    """Sensors that measure the whole farm rather than a single sector."""
    merged = pd.DataFrame()
    for tag in cfg.GLOBAL_SENSORS:
        df = load_sensor(tag, keep_node=False)
        if df is None:
            continue
        merged = df if merged.empty else merged.merge(df, on="ts", how="outer")
    if not merged.empty:
        merged = merged.groupby("ts", as_index=False).mean(numeric_only=True)
    return merged


def load_sector_sensors():
    """Sensors that exist once per sector, keyed by their column name."""
    return {
        cfg.SENSOR_COLUMNS[tag]: df
        for tag in cfg.ZONAL_SENSORS
        if (df := load_sensor(tag, keep_node=True)) is not None
    }


def select_sector(df, column, sector):
    """Keep the readings of one sector and average any co-located probes.

    Soil probes encode the sector in the last character of the node id
    (``SUT00003``); the pH and EC probes encode it before the type suffix
    (``SNPK00003_PH``).
    """
    if column in ("ec", "ph"):
        mask = df["fk_iot_node"].str.contains(f"{sector}_", na=False, regex=False)
    else:
        mask = df["fk_iot_node"].str.endswith(str(sector), na=False)

    selected = df[mask].drop(columns=["fk_iot_node"], errors="ignore")
    if selected.empty:
        return None
    return selected.groupby("ts", as_index=False).mean(numeric_only=True)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def fetch_weather(start, end):
    """Retrieve hourly ERA5 reanalysis for the site and interpolate to minutes."""
    params = {
        "latitude": cfg.LATITUDE,
        "longitude": cfg.LONGITUDE,
        # One day of margin on each side so the interpolation covers the extremes.
        "start_date": (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": list(cfg.WEATHER_VARIABLES),
        "timezone": "auto",
    }
    print(f"  Fetching weather {params['start_date']} -> {params['end_date']}...")

    response = requests.get(cfg.WEATHER_API_URL, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json().get("hourly")
    if not hourly:
        raise RuntimeError("Open-Meteo returned no hourly data.")

    weather = pd.DataFrame(
        {"ts": pd.to_datetime(hourly["time"])}
        | {name: hourly[api_name] for api_name, name in cfg.WEATHER_VARIABLES.items()}
    )
    return weather.set_index("ts").resample("min").interpolate(method="linear").reset_index()


# ---------------------------------------------------------------------------
# Irrigation
# ---------------------------------------------------------------------------
def irrigation_minutes(events, index, interval_start, interval_end):
    """Minutes of valve opening per bin, expanded from the validated events."""
    minutes = pd.Series(
        0, index=pd.date_range(start=interval_start, end=interval_end, freq="min")
    )
    for _, event in events.iterrows():
        start = pd.to_datetime(event["start"]).floor("min")
        end = pd.to_datetime(event["end"]).ceil("min")
        minutes.loc[(minutes.index >= start) & (minutes.index <= end)] = 1

    per_bin = minutes.resample(cfg.SAMPLING_INTERVAL).sum()
    # A bin cannot hold more open minutes than its own length; overlapping
    # events would otherwise push the count above the interval.
    return per_bin.reindex(index).fillna(0).clip(upper=float(cfg.SAMPLING_RATE_MINUTES))


def main():
    cfg.MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sensors...")
    farm_wide = load_farm_wide_sensors()
    per_sector = load_sector_sensors()

    timestamps = [farm_wide["ts"]] if not farm_wide.empty else []
    timestamps += [df["ts"] for df in per_sector.values()]
    if not timestamps:
        raise SystemExit("No sensor data found — check data/original/raw_sensor/.")
    all_ts = pd.concat(timestamps)

    print("Loading weather...")
    weather = fetch_weather(all_ts.min(), all_ts.max())
    farm_wide = (
        weather if farm_wide.empty else farm_wide.merge(weather, on="ts", how="outer")
    ).sort_values("ts")

    print(f"Loading irrigation events from {cfg.EVENTS_FILE.name}...")
    events = pd.read_csv(cfg.EVENTS_FILE, parse_dates=["start", "end"])
    print(f"  {len(events)} events.")

    for sector in cfg.SECTORS:
        print(f"--- Sector {sector} ---")
        table = farm_wide.copy()

        for column, df in per_sector.items():
            selected = select_sector(df, column, sector)
            if selected is not None:
                table = selected if table.empty else table.merge(selected, on="ts", how="outer")

        if table.empty:
            print("  No data, skipping.")
            continue

        table = table.set_index("ts").sort_index()
        resampled = table.resample(cfg.SAMPLING_INTERVAL).mean()

        sector_events = events[events["sector"] == sector]
        if sector_events.empty:
            print("  No irrigation events for this sector.")
            resampled["irrigation_duration_minutes"] = 0.0
            resampled["liters_total"] = 0.0
        else:
            rate = cfg.flow_rate(sector)
            print(f"  Reconstructing irrigation signal at {rate} L/min...")
            resampled["irrigation_duration_minutes"] = irrigation_minutes(
                sector_events, resampled.index, table.index.min(), table.index.max()
            )
            resampled["liters_total"] = resampled["irrigation_duration_minutes"] * rate

        path = cfg.MERGED_DIR / f"dataset_zone_{sector}.csv"
        resampled.reset_index().to_csv(path, index=False)
        print(f"  Saved {path.relative_to(cfg.ROOT)} ({len(resampled)} rows)")

    print("\nStage 01 complete.")


if __name__ == "__main__":
    main()

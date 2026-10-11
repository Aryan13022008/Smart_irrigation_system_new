"""
Stage 00 — irrigation event reconstruction.

Turns raw valve telemetry into a list of validated irrigation and fertigation
events.

The platform records both physically observed valve transitions (``is_raw`` =
True) and synthetic states it infers when no transition was reported
(``is_raw`` = False). Synthetic openings cannot be trusted on their own, so an
event is only accepted when it is also agronomically plausible: the soil was dry
enough to justify irrigating, or a fertigation cycle was running.

Inputs
    data/original/raw_irrigation_actuators/raw_actuators.csv
    data/original/raw_sensor/MEASUREMENT_sut.csv

Outputs
    data/irrigation_actuators/final_irrigation_events.csv
    data/irrigation_actuators/sector_statistics_report.csv

Run from anywhere:  python 00_irrigation_actuators_cleaning.py
"""

import re

import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _trailing_number(node_id):
    """Sector number encoded in the trailing digits of a node identifier."""
    match = re.search(r"(\d+)$", str(node_id))
    return int(match.group(1)) if match else None


def load_valve_telemetry():
    """Load valve records, returning the per-sector irrigation signal and the
    set of instants at which the fertigation unit was delivering."""
    raw = pd.read_csv(cfg.RAW_ACTUATORS_FILE, dtype=str, low_memory=False)
    raw.columns = raw.columns.str.strip()

    raw["time"] = pd.to_datetime(raw["time"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.dropna(subset=["time"])

    raw["is_raw"] = (
        raw["is_raw"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    raw["open"] = pd.to_numeric(raw["open"], errors="coerce").fillna(0).astype(int)

    # AIRR* are the per-sector irrigation valves, AFIRR* the shared fertigation unit.
    irrigation = raw[raw["fk_iot_node"].str.startswith("AIRR")].copy()
    fertigation = raw[raw["fk_iot_node"].str.startswith("AFIRR")]

    irrigation["sector"] = irrigation["fk_iot_node"].apply(
        lambda node: int(m.group(1)) if (m := re.search(r"^AIRR.*?(\d+)$", str(node))) else None
    )
    irrigation = irrigation.dropna(subset=["sector"])
    irrigation["sector"] = irrigation["sector"].astype(int)

    # Several channels can report at the same instant; the valve is open if any
    # of them says so.
    irrigation = irrigation.groupby(["sector", "time"], as_index=False).agg(
        {"open": "max", "is_raw": "max"}
    )

    fertigation_times = set(fertigation.loc[fertigation["open"] == 1, "time"])
    return irrigation, fertigation_times


def load_soil_moisture():
    """Load the soil moisture readings used to validate valve activations."""
    measurements = pd.read_csv(
        cfg.RAW_SENSOR_DIR / cfg.SENSOR_FILES["sut"], dtype=str, low_memory=False
    )
    measurements.columns = measurements.columns.str.strip()

    measurements["measurement_date"] = pd.to_datetime(
        measurements["measurement_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    measurements["humidity_percentage"] = pd.to_numeric(
        measurements["humidity_percentage"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    measurements = measurements.dropna(subset=["measurement_date", "humidity_percentage"])

    measurements["sector"] = measurements["fk_iot_node"].apply(_trailing_number)
    measurements = measurements.dropna(subset=["sector"])
    measurements["sector"] = measurements["sector"].astype(int)

    measurements = measurements[["measurement_date", "sector", "humidity_percentage"]]
    # The probe occasionally reports twice for the same instant. merge_asof then
    # keeps whichever duplicate comes last, so the row order has to be pinned
    # down: sort by sector first, then by time. Sorting by time alone would make
    # the retained reading depend on the order of the source file.
    return measurements.sort_values(["sector", "measurement_date"]).sort_values(
        "measurement_date", kind="stable"
    )


def attach_soil_state(valves, moisture):
    """Attach to every valve record the most recent soil moisture reading of the
    same sector, so each activation can be judged against the soil state."""
    return pd.merge_asof(
        valves.sort_values("time"),
        moisture,
        left_on="time",
        right_on="measurement_date",
        by="sector",
        direction="backward",
    )


# ---------------------------------------------------------------------------
# Phase 1 — event generation
# ---------------------------------------------------------------------------
def _event(sector, start, end, start_moisture, end_moisture, reason, is_fertigation):
    return {
        "sector": sector,
        "start": start,
        "end": end,
        "initial_humidity": start_moisture,
        "final_humidity": end_moisture,
        "stop_reason": reason,
        "type": "fertigation" if is_fertigation else "standard_irrigation",
    }


def _activation_is_valid(rows, i, threshold_start, fertigation_times):
    """Decide whether a synthetic valve opening should be believed.

    It is accepted when the soil is dry enough to justify irrigating or a
    fertigation cycle is running. When neither holds, the signal is followed
    forward: if the valve stays open until a record that *is* trustworthy, the
    opening is a genuine event reported slightly early. A closure encountered
    first means the whole burst was spurious.
    """
    row = rows[i]
    moisture = row["humidity_percentage"]
    if (pd.notna(moisture) and moisture < threshold_start) or row["time"] in fertigation_times:
        return True

    for following in rows[i + 1:]:
        if following["open"] == 0:
            return False
        next_moisture = following["humidity_percentage"]
        if (
            following["is_raw"]
            or (pd.notna(next_moisture) and next_moisture < threshold_start)
            or following["time"] in fertigation_times
        ):
            return True
    return False


def generate_events(records, fertigation_times):
    """Walk each sector's valve records and emit one event per open/close cycle."""
    events = []

    for sector, group in records.groupby("sector"):
        if sector not in cfg.THRESHOLD_STOP:
            continue

        threshold_start = cfg.THRESHOLD_START.get(sector, 40)
        threshold_stop = cfg.THRESHOLD_STOP[sector]
        rows = group.to_dict("records")

        irrigating = False
        is_fertigation = False
        start_time = None
        start_moisture = None
        # Set after a logical stop while the valve is still physically open, so
        # the same opening is not counted twice.
        awaiting_closure = False

        for i, row in enumerate(rows):
            now = row["time"]
            valve_open = row["open"]
            is_raw = row["is_raw"]
            moisture = row["humidity_percentage"]

            forbidden_time = (
                now.hour >= cfg.FORBIDDEN_HOUR_START or now.hour < cfg.FORBIDDEN_HOUR_END
            )

            # A fertigation cycle lasts a fixed time by design.
            if irrigating and is_fertigation:
                elapsed = (now - start_time).total_seconds() / 60
                if elapsed >= cfg.FERTIGATION_TIMEOUT_MINUTES:
                    events.append(_event(
                        sector, start_time,
                        start_time + pd.Timedelta(minutes=cfg.FERTIGATION_TIMEOUT_MINUTES),
                        start_moisture, moisture, "fertigation_timeout", True,
                    ))
                    irrigating = False
                    start_time = start_moisture = None
                    awaiting_closure = True

            # Irrigation is not allowed outside the daytime window.
            if irrigating and forbidden_time:
                events.append(_event(
                    sector, start_time, now, start_moisture, moisture,
                    "time_restriction", is_fertigation,
                ))
                irrigating = False
                start_time = start_moisture = None
                if valve_open == 1:
                    awaiting_closure = True
                continue

            if not irrigating and forbidden_time and valve_open == 1:
                continue

            if valve_open == 0:
                if irrigating:
                    events.append(_event(
                        sector, start_time, now, start_moisture, moisture,
                        "physical_valve_closure", is_fertigation,
                    ))
                    irrigating = False
                    start_time = start_moisture = None
                awaiting_closure = False

            elif valve_open == 1:
                if awaiting_closure or irrigating:
                    continue
                if not is_raw and not _activation_is_valid(
                    rows, i, threshold_start, fertigation_times
                ):
                    continue
                irrigating = True
                start_time = now
                start_moisture = moisture
                is_fertigation = now in fertigation_times

            # A standard irrigation stops once the soil has taken up enough water.
            if irrigating and not is_fertigation and not is_raw:
                if pd.notna(moisture) and moisture >= threshold_stop:
                    events.append(_event(
                        sector, start_time, now, start_moisture, moisture,
                        "humidity_threshold", False,
                    ))
                    irrigating = False
                    start_time = start_moisture = None
                    awaiting_closure = True

        if irrigating:
            events.append(_event(
                sector, start_time, rows[-1]["time"], start_moisture,
                rows[-1]["humidity_percentage"], "end_of_dataset", is_fertigation,
            ))

    return events


# ---------------------------------------------------------------------------
# Phases 2 to 4 — debouncing and filtering
# ---------------------------------------------------------------------------
def merge_close_events(events):
    """Absorb signal flicker by joining events separated by a very short gap."""
    df = pd.DataFrame(events).sort_values(by=["sector", "start"])
    merged = []

    for _, sector_events in df.groupby("sector"):
        rows = [row.to_dict() for _, row in sector_events.iterrows()]
        current = rows[0]
        for nxt in rows[1:]:
            gap_minutes = (nxt["start"] - current["end"]).total_seconds() / 60
            if gap_minutes <= cfg.MERGE_TOLERANCE_MINUTES:
                current["end"] = nxt["end"]
                current["final_humidity"] = nxt["final_humidity"]
                current["stop_reason"] = nxt["stop_reason"]
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    return merged


def filter_events(events):
    """Drop events too short to be a real irrigation, then those so long that a
    closure signal must have been lost."""
    kept = []
    for event in events:
        duration_seconds = (event["end"] - event["start"]).total_seconds()
        if duration_seconds / 60 >= cfg.MIN_DURATION_MINUTES:
            event["duration_hours"] = round(duration_seconds / 3600, 2)
            kept.append(event)

    df = pd.DataFrame(kept)
    if df.empty:
        return df

    anomalous = (df["duration_hours"] > cfg.MAX_DURATION_HOURS) & (
        df["stop_reason"].isin(cfg.ANOMALOUS_STOP_REASONS)
    )
    print(f"  Removed {int(anomalous.sum())} events longer than "
          f"{cfg.MAX_DURATION_HOURS} h with a lost closure signal.")
    return df[~anomalous].copy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_sector_statistics(events):
    """Per-sector summary of how well the observed irrigation matched the
    agronomic operating band of each sector."""
    report = []

    for sector, group in events.groupby("sector"):
        total = len(group)
        low, high = cfg.THRESHOLD_START.get(sector, 40), cfg.THRESHOLD_STOP.get(sector, 60)

        row = {"sector": sector, "total_events": total}
        for kind, label in (("standard_irrigation", "irrigation"), ("fertigation", "fertigation")):
            subset = group[group["type"] == kind]
            # "Optimal" means the sector was inside its operating band when the
            # valve opened: dry enough to need water, not yet at target.
            in_band = subset["initial_humidity"].between(low, high).sum()
            row[f"total_{label}"] = len(subset)
            row[f"{label}_optimal_start"] = int(in_band)
            row[f"{label}_optimal_start_percent"] = (
                round(in_band / len(subset) * 100, 2) if len(subset) else 0.0
            )

        for hours in (2, 5):
            longer = int((group["duration_hours"] > hours).sum())
            row[f"events_longer_{hours}h"] = longer
            row[f"percent_longer_{hours}h"] = round(longer / total * 100, 2) if total else 0.0

        report.append(row)

    columns = [
        "sector", "total_events", "total_irrigation", "total_fertigation",
        "irrigation_optimal_start", "irrigation_optimal_start_percent",
        "fertigation_optimal_start", "fertigation_optimal_start_percent",
        "events_longer_2h", "percent_longer_2h", "events_longer_5h", "percent_longer_5h",
    ]
    return pd.DataFrame(report)[columns]


def main():
    cfg.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading valve telemetry and soil moisture...")
    valves, fertigation_times = load_valve_telemetry()
    moisture = load_soil_moisture()
    records = attach_soil_state(valves, moisture)

    print("Phase 1: generating events with agronomic validation...")
    events = generate_events(records, fertigation_times)
    if not events:
        raise SystemExit("No irrigation events found — check the input files.")
    print(f"  {len(events)} raw events.")

    print(f"Phase 2: merging events closer than {cfg.MERGE_TOLERANCE_MINUTES} min...")
    events = merge_close_events(events)
    print(f"  {len(events)} events after debouncing.")

    print(f"Phase 3-4: filtering events shorter than {cfg.MIN_DURATION_MINUTES} min "
          f"and anomalous long events...")
    final = filter_events(events)
    print(f"  {len(final)} validated events.")

    final.to_csv(cfg.EVENTS_FILE, index=False)
    print(f"Saved {cfg.EVENTS_FILE.relative_to(cfg.ROOT)}")

    stats = build_sector_statistics(final)
    stats.to_csv(cfg.SECTOR_STATS_FILE, index=False)
    print(f"Saved {cfg.SECTOR_STATS_FILE.relative_to(cfg.ROOT)}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

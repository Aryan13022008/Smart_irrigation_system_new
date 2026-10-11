"""
Dataset Extraction, Continuous Segment Splitting & Leakage-Safe Sequence Building (Phases 2 & 3)
================================================================================================
Extracts genuine 10-minute sensor observations from SQLite, handles probe calibration,
detects collection gaps, splits into continuous segments, and builds leakage-safe (X, y) pairs.

Non-negotiable constraints:
- PUMP_ALLOWED = False.
- Strict 6-feature ordered schema.
- Target: future_relative_moisture_index_60m (60m ahead).
- Never build sequences across gaps.
- Chronological train/val/test splits without target window leakage.
- Zero fabrication: no dummy constants, no imputations.
"""

from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import numpy as np
import pandas as pd

PUMP_ALLOWED = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sensor_history.db"

LIVE_FEATURE_SCHEMA = [
    "soil1_relative_pct",
    "soil2_relative_pct",
    "temperature_c",
    "humidity_pct",
    "hour",
    "month",
]

PROBE_CALIBRATION_REFERENCES = {
    "probe1": {"dry_adc": 2054.0, "wet_adc": 1073.0, "pin": "GPIO 32"},
    "probe2": {"dry_adc": 1906.0, "wet_adc": 1023.0, "pin": "GPIO 33"},
}

TARGET_DEFINITION = "future_relative_moisture_index_60m"
TARGET_UNITS = "% relative index (0-100)"
TARGET_HORIZON_MINUTES = 60
DEFAULT_SEQUENCE_LENGTH = 6  # 6 steps @ 10m = 60m input history
DEFAULT_HORIZON_STEPS = 6    # 6 steps @ 10m = 60m ahead


def calculate_calibrated_probe_index(raw_adc, dry_adc, wet_adc):
    """
    Converts raw 12-bit capacitive ADC count into a calibrated relative moisture index (0.0-100.0%)
    using 2-point reference endpoints:
        Index % = clamp(0.0, 100.0, (dry_adc - raw_adc) / (dry_adc - wet_adc) * 100.0)
    Note: Represents relative capacitive impedance ratio, NOT volumetric water content (VWC).
    """
    if raw_adc is None or dry_adc is None or wet_adc is None:
        return None
    try:
        raw = float(raw_adc)
        dry = float(dry_adc)
        wet = float(wet_adc)
    except (ValueError, TypeError):
        return None

    if dry <= wet:
        return None

    pct = ((dry - raw) / (dry - wet)) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


def extract_continuous_segments(df_obs, interval_seconds=600.0, tolerance_seconds=2.0):
    """
    Partitions a sorted DataFrame of observations into continuous segments where
    consecutive interval_start_epoch differences are strictly within tolerance of interval_seconds.

    Returns
    -------
    list of pd.DataFrame: List of contiguous segments (each with length >= 1).
    """
    if df_obs.empty:
        return []

    epochs = df_obs["interval_start_epoch"].values
    segments = []
    current_indices = [0]

    for i in range(1, len(epochs)):
        diff = epochs[i] - epochs[i - 1]
        if np.isclose(diff, interval_seconds, atol=tolerance_seconds):
            current_indices.append(i)
        else:
            segments.append(df_obs.iloc[current_indices].reset_index(drop=True))
            current_indices = [i]

    if current_indices:
        segments.append(df_obs.iloc[current_indices].reset_index(drop=True))

    return segments


def build_sequences_from_segment(
    segment_df,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    horizon_steps=DEFAULT_HORIZON_STEPS,
    probe_calibrations=None,
):
    """
    Builds sliding window sequences within a single contiguous segment.
    Ensures input window (sequence_length steps) and future target (horizon_steps ahead)
    exist strictly within the segment boundary without extrapolation.

    Returns
    -------
    tuple of (X_list, y_list, metadata_list)
    """
    cal = probe_calibrations or PROBE_CALIBRATION_REFERENCES
    p1_dry = cal["probe1"]["dry_adc"]
    p1_wet = cal["probe1"]["wet_adc"]
    p2_dry = cal["probe2"]["dry_adc"]
    p2_wet = cal["probe2"]["wet_adc"]

    n_rows = len(segment_df)
    total_window_needed = sequence_length + horizon_steps
    if n_rows < total_window_needed:
        return [], [], []

    # Precompute calibrated features for the entire segment
    p1_rel = [calculate_calibrated_probe_index(r, p1_dry, p1_wet) for r in segment_df["soil1_raw_mean"]]
    p2_rel = [calculate_calibrated_probe_index(r, p2_dry, p2_wet) for r in segment_df["soil2_raw_mean"]]
    temps = segment_df["temperature_c_mean"].values
    hums = segment_df["humidity_pct_mean"].values

    # Derive deterministic temporal features from interval_start_utc
    dt_series = pd.to_datetime(segment_df["interval_start_utc"], utc=True)
    hours = dt_series.dt.hour.values
    months = dt_series.dt.month.values

    # Precompute average moisture index for targets
    avg_moisture = []
    for s1, s2 in zip(p1_rel, p2_rel):
        if s1 is not None and s2 is not None:
            avg_moisture.append((s1 + s2) / 2.0)
        elif s1 is not None:
            avg_moisture.append(s1)
        elif s2 is not None:
            avg_moisture.append(s2)
        else:
            avg_moisture.append(np.nan)

    X_list = []
    y_list = []
    meta_list = []

    for i in range(n_rows - total_window_needed + 1):
        target_idx = i + sequence_length - 1 + horizon_steps
        target_val = avg_moisture[target_idx]

        # Require finite target
        if np.isnan(target_val):
            continue

        # Extract 6-step input window
        seq_matrix = []
        valid_input = True
        for step in range(i, i + sequence_length):
            s1 = p1_rel[step]
            s2 = p2_rel[step]
            tc = temps[step]
            hm = hums[step]
            hr = hours[step]
            mo = months[step]

            if s1 is None or s2 is None or np.isnan(tc) or np.isnan(hm):
                valid_input = False
                break

            seq_matrix.append([s1, s2, tc, hm, hr, mo])

        if not valid_input or len(seq_matrix) != sequence_length:
            continue

        X_list.append(np.array(seq_matrix, dtype=np.float32))
        y_list.append(float(target_val))
        meta_list.append({
            "input_start_utc": segment_df["interval_start_utc"].iloc[i],
            "input_end_utc": segment_df["interval_start_utc"].iloc[i + sequence_length - 1],
            "target_utc": segment_df["interval_start_utc"].iloc[target_idx],
            "horizon_minutes": horizon_steps * 10,
        })

    return X_list, y_list, meta_list


def build_dataset_from_sqlite(
    db_path=None,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    horizon_steps=DEFAULT_HORIZON_STEPS,
    probe_calibrations=None,
):
    """
    Reads observations from SQLite, splits into continuous segments, and builds all valid (X, y) pairs.

    Returns
    -------
    dict with keys:
        - X : np.ndarray of shape (N, sequence_length, 6)
        - y : np.ndarray of shape (N,)
        - metadata : list of dicts
        - report : comprehensive dataset audit report
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not path.exists():
        empty_report = {
            "error": f"Database file not found at {path}",
            "observation_count": 0,
            "continuous_segments_count": 0,
            "valid_sequences_count": 0,
        }
        return {"X": np.empty((0, sequence_length, 6)), "y": np.empty((0,)), "metadata": [], "report": empty_report}

    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        df_obs = pd.read_sql_query(
            "SELECT * FROM ten_minute_observations ORDER BY interval_start_epoch ASC",
            conn,
        )
    finally:
        conn.close()

    if df_obs.empty:
        empty_report = {
            "error": "Table ten_minute_observations is empty",
            "observation_count": 0,
            "continuous_segments_count": 0,
            "valid_sequences_count": 0,
        }
        return {"X": np.empty((0, sequence_length, 6)), "y": np.empty((0,)), "metadata": [], "report": empty_report}

    segments = extract_continuous_segments(df_obs)

    all_X = []
    all_y = []
    all_meta = []

    for seg in segments:
        X_s, y_s, meta_s = build_sequences_from_segment(
            seg,
            sequence_length=sequence_length,
            horizon_steps=horizon_steps,
            probe_calibrations=probe_calibrations,
        )
        all_X.extend(X_s)
        all_y.extend(y_s)
        all_meta.extend(meta_s)

    X_arr = np.array(all_X, dtype=np.float32) if all_X else np.empty((0, sequence_length, 6), dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32) if all_y else np.empty((0,), dtype=np.float32)

    # Gap and streak statistics
    epochs = df_obs["interval_start_epoch"].values
    diffs = np.diff(epochs) if len(epochs) > 1 else np.array([])
    gap_count = int(np.sum(diffs > 602.0))
    longest_segment = max(len(s) for s in segments) if segments else 0

    target_std = float(np.std(y_arr)) if len(y_arr) > 1 else 0.0
    target_mean = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0

    report = {
        "observation_count": len(df_obs),
        "continuous_segments_count": len(segments),
        "longest_continuous_streak_intervals": longest_segment,
        "longest_continuous_streak_hours": longest_segment * 10 / 60,
        "gap_count": gap_count,
        "valid_sequences_count": len(X_arr),
        "target_name": TARGET_DEFINITION,
        "target_horizon_minutes": horizon_steps * 10,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_schema": LIVE_FEATURE_SCHEMA,
    }

    return {"X": X_arr, "y": y_arr, "metadata": all_meta, "report": report}


def chronological_split(X, y, metadata, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits sequential dataset strictly chronologically into train, val, and test subsets.
    Does NOT shuffle samples, ensuring future patterns are not leaked into earlier sets.
    """
    n = len(X)
    if n < 3:
        raise ValueError(f"Insufficient samples ({n}) to perform a 3-way chronological split.")

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    # Ensure test gets at least 1 sample if n allows
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - (n_train + n_val)

    X_train, y_train, meta_train = X[:n_train], y[:n_train], metadata[:n_train]
    X_val, y_val, meta_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], metadata[n_train : n_train + n_val]
    X_test, y_test, meta_test = X[n_train + n_val:], y[n_train + n_val:], metadata[n_train + n_val:]

    return {
        "train": {"X": X_train, "y": y_train, "metadata": meta_train},
        "val": {"X": X_val, "y": y_val, "metadata": meta_val},
        "test": {"X": X_test, "y": y_test, "metadata": meta_test},
        "split_counts": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    }

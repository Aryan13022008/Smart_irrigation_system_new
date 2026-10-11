"""
Phase 25 — Real-Time Hardware History Infrastructure & Calibration Logger
========================================================================
Implements:
1. Real-time 10-minute observation persistence (`ten_minute_observations` table).
2. Live hardware observation accumulator that ingests 2.5s serial packets and writes
   deterministic 10-minute records with complete sample counts, variance, and coverage metrics.
3. Calibration log repository (`soil_calibration_logs` table) for recording empirical soil
   moisture calibration measurements (gravimetric core tests, precision TDR cross-checks).
4. Real-time sequence buffer extractor that queries `ten_minute_observations` to verify whether
   a genuine, contiguous 24-step (4-hour) sequence is available for hardware forecasting.
5. Strict safety locks: PUMP_ALLOWED = False. Zero actuator actuation.
"""

from pathlib import Path
from datetime import datetime, timezone
import time
import math
import sys
import sqlite3
import numpy as np
import pandas as pd

# Hard safety lock
PUMP_ALLOWED = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sensor_history.db"

INTERVAL_SECONDS = 600.0  # Exactly 10 minutes


def get_connection(db_path=None):
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.Error:
        pass
    return conn


def init_phase25_tables(db_path=None):
    """
    Initializes Phase 25 persistent tables:
    - `ten_minute_observations`: stores aggregated 10-minute sensor observations.
    - `soil_calibration_logs`: stores empirical soil core calibration measurements.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ten_minute_observations (
                    interval_start_epoch REAL PRIMARY KEY,
                    interval_end_epoch REAL NOT NULL,
                    interval_start_utc TEXT NOT NULL,
                    interval_end_utc TEXT NOT NULL,
                    raw_sample_count INTEGER NOT NULL,
                    time_coverage_seconds REAL NOT NULL,
                    soil1_raw_mean REAL,
                    soil1_raw_std REAL,
                    soil2_raw_mean REAL,
                    soil2_raw_std REAL,
                    temperature_c_mean REAL,
                    temperature_c_valid_count INTEGER NOT NULL,
                    humidity_pct_mean REAL,
                    humidity_pct_valid_count INTEGER NOT NULL,
                    rain_detected_count INTEGER NOT NULL,
                    rain_pin_raw_mean REAL,
                    has_valid_rain_data INTEGER NOT NULL,
                    rain_presence_ratio REAL,
                    water_level_raw_mean REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ten_min_epoch 
                ON ten_minute_observations(interval_start_epoch);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS soil_calibration_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    probe_id TEXT NOT NULL,
                    raw_adc REAL NOT NULL,
                    recorded_at REAL NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    reference_value REAL NOT NULL,
                    reference_unit TEXT NOT NULL,
                    soil_type TEXT,
                    bulk_density_g_cm3 REAL,
                    sample_temperature_c REAL,
                    calibration_method TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calib_probe 
                ON soil_calibration_logs(probe_id);
            """)
    finally:
        conn.close()


def sync_raw_readings_to_ten_minute_observations(db_path=None, interval_seconds=INTERVAL_SECONDS):
    """
    Scans `sensor_readings` in SQLite and synchronizes completed 10-minute buckets into
    `ten_minute_observations`. Ensures idempotency with `INSERT OR REPLACE`.

    Returns
    -------
    dict: Summary of sync operation (total_processed, buckets_written, current_open_bucket).
    """
    init_phase25_tables(db_path)
    conn = get_connection(db_path)
    try:
        df_raw = pd.read_sql_query(
            "SELECT recorded_at, soil1_raw, soil2_raw, water_level_raw, rain_pin_raw, "
            "rain_detected, temperature_c, humidity_pct FROM sensor_readings "
            "WHERE status = 'valid' ORDER BY recorded_at ASC",
            conn,
        )
        if df_raw.empty:
            return {"total_processed": 0, "buckets_written": 0, "open_bucket": None}

        # Deterministic bucket floor
        epochs = df_raw["recorded_at"].values.astype(float)
        df_raw["bucket_start"] = (np.floor(epochs / interval_seconds) * interval_seconds).astype(float)

        now_epoch = time.time()
        current_open_bucket = math.floor(now_epoch / interval_seconds) * interval_seconds

        buckets_to_persist = []
        for bucket_epoch, group in df_raw.groupby("bucket_start"):
            bucket_end_epoch = bucket_epoch + interval_seconds
            sample_count = len(group)

            first_t = float(group["recorded_at"].iloc[0])
            last_t = float(group["recorded_at"].iloc[-1])
            time_coverage = float(last_t - first_t)

            start_dt = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc).isoformat()
            end_dt = datetime.fromtimestamp(bucket_end_epoch, tz=timezone.utc).isoformat()

            s1 = group["soil1_raw"].dropna().astype(float)
            s1_mean = float(s1.mean()) if not s1.empty else None
            s1_std = float(s1.std()) if len(s1) > 1 else (0.0 if len(s1) == 1 else None)

            s2 = group["soil2_raw"].dropna().astype(float)
            s2_mean = float(s2.mean()) if not s2.empty else None
            s2_std = float(s2.std()) if len(s2) > 1 else (0.0 if len(s2) == 1 else None)

            tc = group["temperature_c"].dropna().astype(float)
            tc_mean = float(tc.mean()) if not tc.empty else None
            tc_count = int(len(tc))

            hum = group["humidity_pct"].dropna().astype(float)
            hum_mean = float(hum.mean()) if not hum.empty else None
            hum_count = int(len(hum))

            rain_series = group["rain_detected"].dropna()
            has_rain_data = 1 if len(rain_series) > 0 else 0
            rain_count = int(np.sum(rain_series.astype(int) == 1)) if has_rain_data else 0
            rain_ratio = float(np.mean(rain_series.astype(int) == 1)) if has_rain_data else None

            r_pin = group["rain_pin_raw"].dropna().astype(float)
            r_pin_mean = float(r_pin.mean()) if not r_pin.empty else None

            wl = group["water_level_raw"].dropna().astype(float)
            wl_mean = float(wl.mean()) if not wl.empty else None

            status = "completed" if bucket_end_epoch <= now_epoch else "accumulating"

            buckets_to_persist.append((
                bucket_epoch, bucket_end_epoch, start_dt, end_dt,
                sample_count, time_coverage,
                s1_mean, s1_std, s2_mean, s2_std,
                tc_mean, tc_count, hum_mean, hum_count,
                rain_count, r_pin_mean, has_rain_data, rain_ratio,
                wl_mean, status, datetime.now(timezone.utc).isoformat()
            ))

        with conn:
            conn.executemany("""
                INSERT OR REPLACE INTO ten_minute_observations (
                    interval_start_epoch, interval_end_epoch, interval_start_utc, interval_end_utc,
                    raw_sample_count, time_coverage_seconds,
                    soil1_raw_mean, soil1_raw_std, soil2_raw_mean, soil2_raw_std,
                    temperature_c_mean, temperature_c_valid_count,
                    humidity_pct_mean, humidity_pct_valid_count,
                    rain_detected_count, rain_pin_raw_mean, has_valid_rain_data, rain_presence_ratio,
                    water_level_raw_mean, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, buckets_to_persist)

        return {
            "total_processed": len(df_raw),
            "buckets_written": len(buckets_to_persist),
            "current_open_bucket": current_open_bucket,
        }
    finally:
        conn.close()


def log_soil_calibration_measurement(record, db_path=None):
    """
    Saves an empirical soil calibration record into `soil_calibration_logs`.
    Validates mandatory laboratory fields.
    """
    sys_path = Path(__file__).resolve().parents[1] / "phase24"
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from data_collection import validate_calibration_record

    ok, sanitized, err = validate_calibration_record(record)
    if not ok:
        return False, err

    init_phase25_tables(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO soil_calibration_logs (
                    probe_id, raw_adc, recorded_at, timestamp_utc, sample_id,
                    reference_value, reference_unit, soil_type, bulk_density_g_cm3,
                    sample_temperature_c, calibration_method, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                sanitized["probe_id"], sanitized["raw_adc"], sanitized["recorded_at"],
                sanitized["timestamp_utc"], sanitized["sample_id"], sanitized["reference_value"],
                sanitized["reference_unit"], sanitized["soil_type"], sanitized["bulk_density_g_cm3"],
                sanitized["sample_temperature_c"], sanitized["calibration_method"],
                sanitized["notes"], datetime.now(timezone.utc).isoformat()
            ))
        return True, None
    except sqlite3.Error as exc:
        return False, f"Database error saving calibration: {exc}"
    finally:
        conn.close()


def extract_realtime_24step_hardware_sequence(db_path=None, interval_seconds=INTERVAL_SECONDS):
    """
    Queries `ten_minute_observations` to find the most recent continuous 24-step (4-hour)
    historical sequence of real ESP32 observations.

    Returns
    -------
    dict:
      - 'available': bool
      - 'streak_length': int
      - 'sequence_df': pd.DataFrame or None
      - 'blocker': str or None
    """
    init_phase25_tables(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(ten_minute_observations);")
        table_cols = {row["name"] for row in cur.fetchall()}
        if not table_cols:
            return {
                "available": False,
                "streak_length": 0,
                "sequence_df": None,
                "blocker": "No 10-minute observation records in database.",
            }

        desired_cols = [
            "interval_start_epoch",
            "interval_end_epoch",
            "interval_start_utc",
            "interval_end_utc",
            "raw_sample_count",
            "time_coverage_seconds",
            "soil1_raw_mean",
            "soil2_raw_mean",
            "temperature_c_mean",
            "humidity_pct_mean",
            "rain_presence_ratio",
        ]
        select_cols = [c for c in desired_cols if c in table_cols]
        if "interval_start_epoch" not in select_cols:
            select_cols = list(table_cols)

        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM ten_minute_observations ORDER BY interval_start_epoch ASC",
            conn,
        )
        if df.empty:
            return {
                "available": False,
                "streak_length": 0,
                "sequence_df": None,
                "blocker": "No 10-minute observation records in database.",
            }

        # Ensure timestamp columns exist consistently even if schema omitted interval_end_utc
        if "interval_end_utc" not in df.columns:
            if "interval_end_epoch" in df.columns:
                df["interval_end_utc"] = [
                    datetime.fromtimestamp(float(ep), tz=timezone.utc).isoformat() if pd.notna(ep) else None
                    for ep in df["interval_end_epoch"]
                ]
            elif "interval_start_epoch" in df.columns:
                df["interval_end_utc"] = [
                    datetime.fromtimestamp(float(ep) + interval_seconds, tz=timezone.utc).isoformat() if pd.notna(ep) else None
                    for ep in df["interval_start_epoch"]
                ]
            else:
                df["interval_end_utc"] = None

        if "interval_start_utc" not in df.columns and "interval_start_epoch" in df.columns:
            df["interval_start_utc"] = [
                datetime.fromtimestamp(float(ep), tz=timezone.utc).isoformat() if pd.notna(ep) else None
                for ep in df["interval_start_epoch"]
            ]

        # Check consecutive streaks
        epochs = df["interval_start_epoch"].values
        diffs = np.diff(epochs)

        # Walk backward from the latest bucket to find contiguous streak
        streak = [len(df) - 1]
        for i in range(len(df) - 2, -1, -1):
            if np.isclose(epochs[i + 1] - epochs[i], interval_seconds, atol=1.0):
                streak.append(i)
            else:
                break

        streak_length = len(streak)
        if streak_length < 24:
            return {
                "available": False,
                "streak_length": streak_length,
                "sequence_df": None,
                "blocker": (
                    f"Latest contiguous streak has {streak_length} observations ({streak_length * 10 / 60:.1f} hours). "
                    f"Sequence model requires 24 continuous 10-minute observations (4.0 hours)."
                ),
            }

        # Select the 24 trailing steps of the streak
        seq_indices = streak[:24][::-1]
        seq_df = df.iloc[seq_indices].copy().reset_index(drop=True)

        return {
            "available": True,
            "streak_length": streak_length,
            "sequence_df": seq_df,
            "blocker": None,
        }
    finally:
        conn.close()

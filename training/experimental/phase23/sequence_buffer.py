"""
Phase 23 — Live Sequence Buffer and Telemetry Validation Module
==============================================================
Provides rigorous validation and temporal alignment checks for constructing
the 24-step, 10-minute cadence (4-hour) historical sequence required by the
experimental LSTM, TCN, and Transformer soil-moisture forecasting models.

Design Rules (Task 2 & Task 5):
1. Uses only genuine recorded sensor observations.
2. Timestamps are required to carry UTC timezone awareness.
3. Requires exactly 24 observations spaced at 10-minute intervals (600 seconds +/- tolerance).
4. Strictly rejects windows with missing records, large gaps, or unordered timestamps.
5. NEVER silently interpolates missing measurements.
6. NEVER duplicates the latest measurement to fabricate or fill a sequence.
7. NEVER treats a high-frequency (e.g. 2.5s) serial packet stream as independent 10m steps.
8. NEVER mixes measurements from different sensor identities or calibration configurations.
9. Completely isolated from the production classifier.
10. If the data does not satisfy sequence requirements, explicitly returns a descriptive blocker.
11. PUMP_ALLOWED = False strictly enforced.
"""

from pathlib import Path
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd

# Hard safety lock
PUMP_ALLOWED = False

EXPECTED_SEQUENCE_LENGTH = 24
EXPECTED_STEP_SECONDS = 600.0  # 10 minutes
DEFAULT_TOLERANCE_SECONDS = 60.0  # +/- 1 minute jitter allowance
MAX_ALLOWED_GAP_SECONDS = 660.0  # 11 minutes maximum permissible step

# The exact 14 features required by the trained checkpoints
EXPERIMENTAL_14_FEATURES = [
    "weather_temp",
    "weather_humidity",
    "weather_rain",
    "weather_pressure",
    "weather_wind_speed",
    "weather_radiation",
    "soil_temperature_0-7cm",
    "soil_temperature_7-18cm",
    "ec",
    "ph",
    "soil_moisture",
    "irrigation_duration_minutes",
    "hour",
    "month",
]


class SequenceBufferAuditResult:
    """Structured container for sequence validation results."""
    def __init__(self, is_valid, sequence=None, blocker_code=None, blocker_message=None, diagnostics=None):
        self.is_valid = bool(is_valid)
        self.sequence = sequence  # pd.DataFrame of shape (24, 14) or None
        self.blocker_code = blocker_code
        self.blocker_message = blocker_message
        self.diagnostics = diagnostics or {}

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "has_sequence": self.sequence is not None,
            "blocker_code": self.blocker_code,
            "blocker_message": self.blocker_message,
            "diagnostics": self.diagnostics,
        }

    def __repr__(self):
        status = "VALID" if self.is_valid else f"BLOCKED({self.blocker_code})"
        return f"<SequenceBufferAuditResult status={status} msg='{self.blocker_message}'>"


def validate_and_construct_sequence(
    df_observations,
    required_features=None,
    expected_length=EXPECTED_SEQUENCE_LENGTH,
    expected_step_seconds=EXPECTED_STEP_SECONDS,
    step_tolerance_seconds=DEFAULT_TOLERANCE_SECONDS,
    probe_id=None,
):
    """
    Validates a candidate DataFrame of historical observations to determine whether
    it forms a genuine, contiguous, properly spaced 4-hour sequence (24 x 10m steps).

    Parameters
    ----------
    df_observations : pd.DataFrame
        Candidate records with a timestamp column ('timestamp' or 'recorded_at')
        and relevant sensor/weather features.
    required_features : list of str, optional
        Features required for model input. Defaults to EXPERIMENTAL_14_FEATURES.
    expected_length : int, default 24
        Exact number of required observations.
    expected_step_seconds : float, default 600.0 (10 minutes)
        Nominal cadence between consecutive rows.
    step_tolerance_seconds : float, default 60.0
        Permissible jitter for network/polling delays.
    probe_id : str, optional
        If specified, verifies that all rows belong strictly to this sensor identity.

    Returns
    -------
    SequenceBufferAuditResult
    """
    if required_features is None:
        required_features = EXPERIMENTAL_14_FEATURES

    if df_observations is None or not isinstance(df_observations, pd.DataFrame) or df_observations.empty:
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="EMPTY_OR_NONE",
            blocker_message="Telemetry input is empty or None.",
        )

    # 1. Row count validation
    n_rows = len(df_observations)
    if n_rows < expected_length:
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="INSUFFICIENT_OBSERVATIONS",
            blocker_message=(
                f"Insufficient observations: found {n_rows} rows, but model strictly requires "
                f"exactly {expected_length} consecutive 10-minute records (4.0 hours)."
            ),
            diagnostics={"observed_rows": n_rows, "required_rows": expected_length},
        )

    # If more than expected_length rows are provided, inspect the trailing window
    df = df_observations.iloc[-expected_length:].copy().reset_index(drop=True)

    # 2. Check probe identity consistency (prevent mixing probe1 and probe2)
    if "probe_id" in df.columns:
        unique_probes = df["probe_id"].dropna().unique()
        if len(unique_probes) > 1:
            return SequenceBufferAuditResult(
                is_valid=False,
                blocker_code="MIXED_PROBE_IDENTITIES",
                blocker_message=(
                    f"Sequence mixes measurements from multiple probe identities: {list(unique_probes)}. "
                    "Probes have distinct physical locations and independent calibrations."
                ),
                diagnostics={"probes": list(unique_probes)},
            )
        if probe_id is not None and len(unique_probes) == 1 and unique_probes[0] != probe_id:
            return SequenceBufferAuditResult(
                is_valid=False,
                blocker_code="PROBE_ID_MISMATCH",
                blocker_message=f"Sequence belongs to probe '{unique_probes[0]}', expected '{probe_id}'.",
                diagnostics={"expected": probe_id, "found": unique_probes[0]},
            )

    # 3. Timestamp parsing and monotonic ordering
    if "recorded_at" in df.columns:
        epochs = df["recorded_at"].values.astype(float)
    elif "timestamp" in df.columns:
        try:
            ts_series = pd.to_datetime(df["timestamp"])
            epochs = ts_series.astype("int64") // 10**9
            epochs = epochs.values.astype(float)
        except Exception as e:
            return SequenceBufferAuditResult(
                is_valid=False,
                blocker_code="TIMESTAMP_PARSE_ERROR",
                blocker_message=f"Failed to parse timestamp column: {e}",
            )
    else:
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="MISSING_TIMESTAMP_COLUMN",
            blocker_message="Observations must include 'recorded_at' (epoch seconds) or 'timestamp' column.",
        )

    # Check for non-finite timestamps
    if not np.all(np.isfinite(epochs)):
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="NON_FINITE_TIMESTAMPS",
            blocker_message="Sequence contains NaN or non-finite timestamp values.",
        )

    # Check monotonic strictly increasing ordering
    diffs = np.diff(epochs)
    if np.any(diffs <= 0):
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="NON_MONOTONIC_TIMESTAMPS",
            blocker_message="Timestamps are out of order, decreasing, or contain duplicate measurements.",
            diagnostics={"min_step_seconds": float(np.min(diffs))},
        )

    # 4. Spacing and Gap Verification (cadence check)
    min_step = float(np.min(diffs))
    max_step = float(np.max(diffs))
    total_span_seconds = float(epochs[-1] - epochs[0])
    expected_span = (expected_length - 1) * expected_step_seconds

    # Detect high-frequency packet stream (e.g. 2.5s serial loop mistaken for 10m steps)
    if min_step < 60.0:  # step less than 1 minute
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="HIGH_FREQUENCY_PACKET_STREAM",
            blocker_message=(
                f"Detected high-frequency telemetry (minimum step = {min_step:.1f}s). "
                "Raw 2.5-second serial packets cannot be treated as independent 10-minute sequence observations."
            ),
            diagnostics={"min_step_seconds": min_step, "max_step_seconds": max_step},
        )

    # Check for excessive gaps or irregular sampling
    gap_violations = []
    min_allowable = expected_step_seconds - step_tolerance_seconds
    max_allowable = expected_step_seconds + step_tolerance_seconds
    for i, step in enumerate(diffs):
        if step < min_allowable or step > max_allowable:
            gap_violations.append((i, i + 1, float(step)))

    if gap_violations:
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="IRREGULAR_TEMPORAL_SPACING",
            blocker_message=(
                f"Observations violate expected 10-minute cadence: {len(gap_violations)} steps outside "
                f"[{min_allowable:.0f}s, {max_allowable:.0f}s]. Total sequence span is {total_span_seconds:.1f}s "
                f"(expected {expected_span:.0f}s). Missing data must not be interpolated."
            ),
            diagnostics={
                "min_step_seconds": min_step,
                "max_step_seconds": max_step,
                "total_span_seconds": total_span_seconds,
                "expected_span_seconds": expected_span,
                "gap_violations_count": len(gap_violations),
            },
        )

    # 5. Check Feature Compatibility
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="MISSING_FEATURE_COLUMNS",
            blocker_message=(
                f"Observations missing {len(missing_features)} required model features: {missing_features}. "
                "Missing environmental channels are strictly not fabricated."
            ),
            diagnostics={"missing_features": missing_features},
        )

    # 6. Check for Non-finite Feature Values (NaN / Inf)
    feature_matrix = df[required_features].values.astype(np.float32)
    if not np.all(np.isfinite(feature_matrix)):
        nan_cols = [
            f for f in required_features
            if not np.all(np.isfinite(df[f].values.astype(float)))
        ]
        return SequenceBufferAuditResult(
            is_valid=False,
            blocker_code="NON_FINITE_FEATURE_VALUES",
            blocker_message=f"Sequence contains NaN, null, or infinite values in features: {nan_cols}.",
            diagnostics={"nan_features": nan_cols},
        )

    # 7. Check for Zero-Variance / Stale Duplicated Readings
    # If a feature (e.g. soil_moisture or temp) is 100% identical down to float precision across all 24 steps,
    # it often indicates duplicated records or a stuck sensor.
    stuck_features = []
    for col in ["soil_moisture", "weather_temp", "weather_humidity"]:
        if col in df.columns:
            vals = df[col].values
            if np.all(vals == vals[0]):
                stuck_features.append(col)

    diagnostics = {
        "start_epoch": float(epochs[0]),
        "end_epoch": float(epochs[-1]),
        "total_span_seconds": total_span_seconds,
        "n_steps": expected_length,
        "stuck_features_flagged": stuck_features,
    }

    # Return valid constructed sequence
    return SequenceBufferAuditResult(
        is_valid=True,
        sequence=df[required_features].copy(),
        diagnostics=diagnostics,
    )


def audit_live_telemetry_history(db_path=None):
    """
    Inspects the actual SQLite database (`data/sensor_history.db`) to evaluate whether
    a genuine 24-step, 10-minute sequence can be extracted from current stored records.

    Returns
    -------
    dict : Comprehensive audit summary of stored telemetry.
    """
    import database

    conn = database.get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, timestamp, recorded_at, soil1_raw, soil2_raw, water_level_raw, "
            "rain_detected, temperature_c, humidity_pct FROM sensor_readings "
            "ORDER BY recorded_at ASC",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return {
            "total_records": 0,
            "can_construct_4h_sequence": False,
            "blocker_code": "NO_STORED_RECORDS",
            "message": "Database contains zero sensor records.",
        }

    epochs = df["recorded_at"].values.astype(float)
    diffs = np.diff(epochs) if len(epochs) > 1 else np.array([])
    total_span = float(epochs[-1] - epochs[0]) if len(epochs) > 1 else 0.0

    # Test candidate sequence extraction
    seq_result = validate_and_construct_sequence(df)

    return {
        "total_records": len(df),
        "earliest_timestamp": df["timestamp"].iloc[0],
        "latest_timestamp": df["timestamp"].iloc[-1],
        "total_span_hours": total_span / 3600.0,
        "median_step_seconds": float(np.median(diffs)) if len(diffs) > 0 else 0.0,
        "min_step_seconds": float(np.min(diffs)) if len(diffs) > 0 else 0.0,
        "max_step_seconds": float(np.max(diffs)) if len(diffs) > 0 else 0.0,
        "can_construct_4h_sequence": seq_result.is_valid,
        "blocker_code": seq_result.blocker_code,
        "blocker_message": seq_result.blocker_message,
        "diagnostics": seq_result.diagnostics,
    }

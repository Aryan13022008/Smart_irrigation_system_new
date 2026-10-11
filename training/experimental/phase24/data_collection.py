"""
Phase 24 — Real-World Sensor Data Collection and Calibration Pipeline Module
============================================================================
Provides deterministic 10-minute observation aggregation from raw ESP32 telemetry,
calibration-record validation and formatting for soil probes, and comprehensive
14-feature live readiness reporting.

Safety Constraints:
- PUMP_ALLOWED = False strictly enforced. Zero actuator control.
- Never fabricates missing sensor measurements.
- Never claims uncalibrated ADC values or relative indices are volumetric water content (VWC).
- Preserves all raw database records without destructive modification.
- Isolated from the production 11-feature classifier.
"""

from pathlib import Path
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd

PUMP_ALLOWED = False

AGGREGATION_INTERVAL_SECONDS = 600.0  # Exactly 10 minutes (600s)

# Exact 14 features expected by trained experimental models (LSTM, TCN, Transformer)
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


# ============================================================================
# TASK 2: TEN-MINUTE OBSERVATION AGGREGATION
# ============================================================================

def aggregate_raw_telemetry(df_raw, interval_seconds=AGGREGATION_INTERVAL_SECONDS):
    """
    Deterministically aggregates timestamped raw ESP32 serial telemetry into
    10-minute observation records aligned to floor(epoch / interval_seconds) * interval_seconds.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw telemetry records from SQLite `sensor_readings`. Expected columns:
        'recorded_at' (float epoch), 'timestamp' (ISO string),
        'soil1_raw', 'soil2_raw', 'water_level_raw', 'rain_pin_raw',
        'rain_detected', 'temperature_c', 'humidity_pct'.
    interval_seconds : float, default 600.0 (10 minutes)
        Aggregation bucket duration.

    Returns
    -------
    pd.DataFrame
        Aggregated 10-minute records. If df_raw is empty, returns empty DataFrame with schema.
    """
    columns = [
        "interval_start_epoch",
        "interval_end_epoch",
        "interval_start_utc",
        "interval_end_utc",
        "raw_sample_count",
        "time_coverage_seconds",
        "soil1_raw_mean",
        "soil1_raw_std",
        "soil2_raw_mean",
        "soil2_raw_std",
        "temperature_c_mean",
        "temperature_c_valid_count",
        "humidity_pct_mean",
        "humidity_pct_valid_count",
        "rain_detected_count",
        "rain_pin_raw_mean",
        "has_valid_rain_data",
        "rain_presence_ratio",
        "water_level_raw_mean",
        "status",
    ]

    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame(columns=columns)

    if "recorded_at" not in df_raw.columns:
        raise ValueError("Raw telemetry must contain 'recorded_at' epoch float column.")

    # Drop any non-finite epoch rows
    df = df_raw[np.isfinite(df_raw["recorded_at"].values.astype(float))].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["recorded_at"] = df["recorded_at"].astype(float)
    df = df.sort_values("recorded_at").reset_index(drop=True)

    # Deterministic interval boundary: floor(epoch / interval_seconds) * interval_seconds
    df["bucket_start_epoch"] = (np.floor(df["recorded_at"] / interval_seconds) * interval_seconds).astype(float)

    aggregated_rows = []
    for bucket_epoch, group in df.groupby("bucket_start_epoch"):
        bucket_end_epoch = bucket_epoch + interval_seconds
        sample_count = len(group)

        # Time coverage: span between first and last reading inside bucket
        first_t = float(group["recorded_at"].iloc[0])
        last_t = float(group["recorded_at"].iloc[-1])
        time_coverage = float(last_t - first_t)

        start_dt = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc).isoformat()
        end_dt = datetime.fromtimestamp(bucket_end_epoch, tz=timezone.utc).isoformat()

        # Soil 1 & 2
        s1 = group["soil1_raw"].dropna().astype(float) if "soil1_raw" in group.columns else pd.Series(dtype=float)
        s1_mean = float(s1.mean()) if not s1.empty else None
        s1_std = float(s1.std()) if len(s1) > 1 else (0.0 if len(s1) == 1 else None)

        s2 = group["soil2_raw"].dropna().astype(float) if "soil2_raw" in group.columns else pd.Series(dtype=float)
        s2_mean = float(s2.mean()) if not s2.empty else None
        s2_std = float(s2.std()) if len(s2) > 1 else (0.0 if len(s2) == 1 else None)

        # DHT22 Temperature & Humidity
        tc = group["temperature_c"].dropna().astype(float) if "temperature_c" in group.columns else pd.Series(dtype=float)
        tc_mean = float(tc.mean()) if not tc.empty else None
        tc_count = int(len(tc))

        hum = group["humidity_pct"].dropna().astype(float) if "humidity_pct" in group.columns else pd.Series(dtype=float)
        hum_mean = float(hum.mean()) if not hum.empty else None
        hum_count = int(len(hum))

        # Rain Sensor (Binary presence vs Depth distinction)
        # rain_detected: 1 if wet/rain, 0 if dry, None if invalid/missing
        rain_series = group["rain_detected"].dropna() if "rain_detected" in group.columns else pd.Series(dtype=float)
        has_rain_data = len(rain_series) > 0
        rain_detected_count = int(np.sum(rain_series.astype(int) == 1)) if has_rain_data else 0
        rain_ratio = float(np.mean(rain_series.astype(int) == 1)) if has_rain_data else None

        r_pin = group["rain_pin_raw"].dropna().astype(float) if "rain_pin_raw" in group.columns else pd.Series(dtype=float)
        r_pin_mean = float(r_pin.mean()) if not r_pin.empty else None

        # Water level raw (reservoir level, strictly NOT irrigation volume)
        wl = group["water_level_raw"].dropna().astype(float) if "water_level_raw" in group.columns else pd.Series(dtype=float)
        wl_mean = float(wl.mean()) if not wl.empty else None

        row = {
            "interval_start_epoch": bucket_epoch,
            "interval_end_epoch": bucket_end_epoch,
            "interval_start_utc": start_dt,
            "interval_end_utc": end_dt,
            "raw_sample_count": sample_count,
            "time_coverage_seconds": time_coverage,
            "soil1_raw_mean": s1_mean,
            "soil1_raw_std": s1_std,
            "soil2_raw_mean": s2_mean,
            "soil2_raw_std": s2_std,
            "temperature_c_mean": tc_mean,
            "temperature_c_valid_count": tc_count,
            "humidity_pct_mean": hum_mean,
            "humidity_pct_valid_count": hum_count,
            "rain_detected_count": rain_detected_count,
            "rain_pin_raw_mean": r_pin_mean,
            "has_valid_rain_data": has_rain_data,
            "rain_presence_ratio": rain_ratio,
            "water_level_raw_mean": wl_mean,
            "status": "valid" if (sample_count > 0 and (s1_mean is not None or s2_mean is not None)) else "incomplete",
        }
        aggregated_rows.append(row)

    res_df = pd.DataFrame(aggregated_rows)
    return res_df.sort_values("interval_start_epoch").reset_index(drop=True)


def identify_aggregation_gaps(aggregated_df, interval_seconds=AGGREGATION_INTERVAL_SECONDS):
    """
    Detects missing 10-minute intervals and disconnect gaps in aggregated observations.
    Never interpolates across missing intervals.
    """
    if aggregated_df.empty or len(aggregated_df) < 2:
        return []

    starts = aggregated_df["interval_start_epoch"].values
    gaps = []
    for i in range(len(starts) - 1):
        actual_step = starts[i + 1] - starts[i]
        if actual_step > interval_seconds:
            missing_count = int(round((actual_step - interval_seconds) / interval_seconds))
            gaps.append({
                "after_interval_utc": aggregated_df["interval_start_utc"].iloc[i],
                "before_interval_utc": aggregated_df["interval_start_utc"].iloc[i + 1],
                "gap_seconds": float(actual_step - interval_seconds),
                "missing_intervals_count": missing_count,
            })
    return gaps


# ============================================================================
# TASK 3: CALIBRATION DATA COLLECTION
# ============================================================================

VALID_REFERENCE_UNITS = ["vwc_pct", "vwc_ratio", "gravimetric_pct"]
VALID_PROBE_IDS = ["probe1", "probe2"]

def validate_calibration_record(record):
    """
    Validates a calibration observation record for capacitive soil moisture probes.

    Mandatory fields:
    - probe_id: 'probe1' or 'probe2'
    - raw_adc: float in 12-bit range [0, 4095]
    - timestamp: ISO 8601 string or epoch float
    - sample_id: non-empty string identifier (e.g., 'CORE_01_AIR', 'CORE_01_SATURATED')
    - reference_value: float, ground-truth laboratory measurement
    - reference_unit: must be one of ['vwc_pct', 'vwc_ratio', 'gravimetric_pct']

    Optional fields:
    - soil_type: str (e.g. 'Sandy Loam', 'Clay')
    - bulk_density_g_cm3: float (> 0)
    - sample_temperature_c: float
    - calibration_method: str (e.g. 'gravimetric_oven_dry', 'precision_tdr_bench')
    - notes: str

    Returns
    -------
    (bool, dict, str): (is_valid, sanitized_record, error_message)
    """
    if not isinstance(record, dict):
        return False, None, "Calibration record must be a dictionary."

    # 1. Probe identity
    probe_id = record.get("probe_id")
    if probe_id not in VALID_PROBE_IDS:
        return False, None, f"Invalid probe_id '{probe_id}'. Must be 'probe1' or 'probe2'."

    # 2. Raw ADC measurement
    raw_adc = record.get("raw_adc")
    if raw_adc is None:
        return False, None, "Missing mandatory field 'raw_adc'."
    try:
        raw_adc = float(raw_adc)
    except (ValueError, TypeError):
        return False, None, f"raw_adc must be numeric, got {raw_adc}."
    if not (0.0 <= raw_adc <= 4095.0):
        return False, None, f"raw_adc ({raw_adc}) out of 12-bit ADC range [0, 4095]."

    # 3. Timestamp
    ts = record.get("timestamp")
    if ts is None:
        return False, None, "Missing mandatory field 'timestamp'."
    if isinstance(ts, (int, float)):
        recorded_at = float(ts)
        iso_ts = datetime.fromtimestamp(recorded_at, tz=timezone.utc).isoformat()
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            recorded_at = dt.timestamp()
            iso_ts = dt.isoformat()
        except ValueError:
            return False, None, f"Invalid ISO 8601 timestamp string: '{ts}'."
    else:
        return False, None, "Timestamp must be float epoch or ISO 8601 string."

    # 4. Sample Identifier
    sample_id = record.get("sample_id")
    if not sample_id or not str(sample_id).strip():
        return False, None, "Missing or empty mandatory field 'sample_id'."
    sample_id = str(sample_id).strip()

    # 5. Reference moisture measurement & units (Strictly no automatic generation)
    ref_val = record.get("reference_value")
    if ref_val is None:
        return False, None, "Missing mandatory field 'reference_value'. Reference measurements must never be fabricated."
    try:
        ref_val = float(ref_val)
    except (ValueError, TypeError):
        return False, None, f"reference_value must be numeric, got {ref_val}."
    if not np.isfinite(ref_val):
        return False, None, "reference_value must be a finite number."

    ref_unit = record.get("reference_unit")
    if ref_unit not in VALID_REFERENCE_UNITS:
        return False, None, f"Invalid reference_unit '{ref_unit}'. Must be one of {VALID_REFERENCE_UNITS}."

    # Range check on reference values
    if ref_unit in ("vwc_pct", "gravimetric_pct") and not (0.0 <= ref_val <= 100.0):
        return False, None, f"reference_value ({ref_val}) out of percentage range [0.0, 100.0] for unit '{ref_unit}'."
    if ref_unit == "vwc_ratio" and not (0.0 <= ref_val <= 1.0):
        return False, None, f"reference_value ({ref_val}) out of ratio range [0.0, 1.0] for unit 'vwc_ratio'."

    # Optional fields
    sanitized = {
        "probe_id": probe_id,
        "raw_adc": raw_adc,
        "recorded_at": recorded_at,
        "timestamp_utc": iso_ts,
        "sample_id": sample_id,
        "reference_value": ref_val,
        "reference_unit": ref_unit,
        "soil_type": str(record.get("soil_type", "unspecified")),
        "bulk_density_g_cm3": float(record["bulk_density_g_cm3"]) if record.get("bulk_density_g_cm3") is not None else None,
        "sample_temperature_c": float(record["sample_temperature_c"]) if record.get("sample_temperature_c") is not None else None,
        "calibration_method": str(record.get("calibration_method", "manual")),
        "notes": str(record.get("notes", "")),
    }

    return True, sanitized, None


# ============================================================================
# TASK 4: FEATURE-READINESS REPORT GENERATOR
# ============================================================================

FEATURE_AUDIT_SPEC = {
    "weather_temp": {
        "status": "Available from genuine live data",
        "live_source": "DHT22 sensor on GPIO 4",
        "units": "°C",
        "scale_match": True,
        "blocker": "None for single-step. Requires 4h contiguous 10-minute history buffer for sequence models.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "weather_humidity": {
        "status": "Available from genuine live data",
        "live_source": "DHT22 sensor on GPIO 4",
        "units": "%",
        "scale_match": True,
        "blocker": "None for single-step. Requires 4h contiguous 10-minute history buffer for sequence models.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "weather_rain": {
        "status": "Available but incompatible in scale or meaning",
        "live_source": "Digital rain board on GPIO 27 (rain_detected)",
        "units": "Binary (0=dry, 1=wet) vs Training mm/10min",
        "scale_match": False,
        "blocker": "Sensor detects qualitative surface water presence, NOT quantitative rainfall depth (mm).",
        "requires_calibration": True,
        "requires_external": True,
    },
    "weather_pressure": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "hPa (barometric pressure)",
        "scale_match": False,
        "blocker": "No barometric sensor installed on edge station. Zero fabrication policy.",
        "requires_calibration": False,
        "requires_external": True,
    },
    "weather_wind_speed": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "m/s",
        "scale_match": False,
        "blocker": "No anemometer installed on edge station. Zero fabrication policy.",
        "requires_calibration": False,
        "requires_external": True,
    },
    "weather_radiation": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "W/m² (global solar flux)",
        "scale_match": False,
        "blocker": "No pyranometer installed on edge station. Zero fabrication policy.",
        "requires_calibration": False,
        "requires_external": True,
    },
    "soil_temperature_0-7cm": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "°C (topsoil depth)",
        "scale_match": False,
        "blocker": "No buried topsoil thermistor. DHT22 measures ambient air temperature, not soil matrix.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "soil_temperature_7-18cm": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "°C (subsoil depth)",
        "scale_match": False,
        "blocker": "No buried subsoil thermistor. Missing physical channel.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "ec": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "mS/cm (bulk electrical conductivity)",
        "scale_match": False,
        "blocker": "No soil EC probe installed. Missing physical channel.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "ph": {
        "status": "Missing",
        "live_source": "None installed on ESP32",
        "units": "pH units (0-14 scale)",
        "scale_match": False,
        "blocker": "No soil pH probe installed. Missing physical channel.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "soil_moisture": {
        "status": "Requires calibration",
        "live_source": "Capacitive probes (GPIO 32 / GPIO 33)",
        "units": "Relative air/water ADC index (%) vs Training VWC (%)",
        "scale_match": False,
        "blocker": "Probes output relative impedance ratio (0% in air, 100% in water), not volumetric water content (VWC). Requires gravimetric soil core calibration curve.",
        "requires_calibration": True,
        "requires_external": False,
    },
    "irrigation_duration_minutes": {
        "status": "Available but incompatible in scale or meaning",
        "live_source": "water_level_raw (GPIO 34) / Actuator status",
        "units": "Minutes of valve actuation per step",
        "scale_match": False,
        "blocker": "Water level sensor measures reservoir depth, NOT irrigation actuator volume or duration. Pump is hard-locked (PUMP_ALLOWED=False).",
        "requires_calibration": False,
        "requires_external": False,
    },
    "hour": {
        "status": "Available from genuine live data",
        "live_source": "Host system UTC clock",
        "units": "Hour of day (0-23)",
        "scale_match": True,
        "blocker": "Deterministic from timestamp. Requires valid sequence timestamps.",
        "requires_calibration": False,
        "requires_external": False,
    },
    "month": {
        "status": "Available from genuine live data",
        "live_source": "Host system UTC clock",
        "units": "Month of year (1-12)",
        "scale_match": True,
        "blocker": "Deterministic from timestamp. Requires valid sequence timestamps.",
        "requires_calibration": False,
        "requires_external": False,
    },
}

def get_14_feature_readiness_audit():
    """Returns structured readiness summary for all 14 features."""
    rows = []
    category_counts = {
        "Available from genuine live data": 0,
        "Available but incompatible in scale or meaning": 0,
        "Missing": 0,
        "Requires calibration": 0,
    }

    for feat in EXPERIMENTAL_14_FEATURES:
        info = FEATURE_AUDIT_SPEC[feat]
        cat = info["status"]
        if cat in category_counts:
            category_counts[cat] += 1
        rows.append({
            "Feature": feat,
            "Readiness Status": cat,
            "Live Hardware Source": info["live_source"],
            "Units & Representation": info["units"],
            "Scale Compatible?": "✅ Yes" if info["scale_match"] else "❌ No",
            "Specific Blocker / Details": info["blocker"],
        })

    return {
        "total_features": len(EXPERIMENTAL_14_FEATURES),
        "category_counts": category_counts,
        "is_ready_for_live_forecasting": False,
        "mandatory_status_label": "LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL",
        "rows": rows,
    }


def audit_sqlite_data_collection(db_path=None):
    """
    Audits SQLite database data collection and 10-minute coverage status.
    """
    import database
    conn = database.get_connection(db_path)
    try:
        df_raw = pd.read_sql_query(
            "SELECT timestamp, recorded_at, soil1_raw, soil2_raw, water_level_raw, "
            "rain_pin_raw, rain_detected, temperature_c, humidity_pct FROM sensor_readings "
            "ORDER BY recorded_at ASC",
            conn,
        )
        probes = database.get_all_calibrations(db_path)
    finally:
        conn.close()

    if df_raw.empty:
        return {
            "total_raw_samples": 0,
            "earliest_ts": None,
            "latest_ts": None,
            "total_span_hours": 0.0,
            "aggregated_intervals_count": 0,
            "gaps": [],
            "probes_calibration": probes,
            "feature_readiness": get_14_feature_readiness_audit(),
        }

    agg_df = aggregate_raw_telemetry(df_raw)
    gaps = identify_aggregation_gaps(agg_df)

    latest_row = df_raw.iloc[-1].to_dict()
    earliest_epoch = float(df_raw["recorded_at"].iloc[0])
    latest_epoch = float(df_raw["recorded_at"].iloc[-1])

    return {
        "total_raw_samples": len(df_raw),
        "earliest_ts": df_raw["timestamp"].iloc[0],
        "latest_ts": df_raw["timestamp"].iloc[-1],
        "total_span_hours": (latest_epoch - earliest_epoch) / 3600.0,
        "latest_reading": latest_row,
        "aggregated_intervals_count": len(agg_df),
        "gaps": gaps,
        "total_missing_intervals": sum(g["missing_intervals_count"] for g in gaps),
        "probes_calibration": probes,
        "feature_readiness": get_14_feature_readiness_audit(),
    }

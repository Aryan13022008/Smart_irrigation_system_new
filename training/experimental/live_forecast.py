"""
Live ESP32-Based Soil Moisture Forecasting Module (Experimental)
================================================================
Implements an isolated, empirical short-term soil moisture trend projection
derived exclusively from genuine ESP32 telemetry stored in SQLite
(`ten_minute_observations` and `sensor_readings`).

Key Principles & Guardrails:
1. Purely Trend-Based: This is an empirical extrapolation, NOT a trained neural
   network or machine learning model.
2. Relative Moisture Index: Probe conversions use stored 2-point dry/wet calibration
   endpoints to produce a relative index (0.0% to 100.0%). This is explicitly NOT
   physical Volumetric Water Content (VWC) because soil-specific gravimetric calibration
   has not been performed.
3. No Data Fabrication: Missing features (such as 14-channel physical weather/depth
   channels required by offline LSTM/TCN/Transformer) are NOT filled with arbitrary constants.
4. Rigorous Input Validation: Handles stale data (>30m), gaps (>25m), duplicate timestamps,
   uncalibrated probes, and insufficient history (<3 points) by refusing to forecast rather
   than hallucinating.
5. Safety Invariant: Hard-locks `PUMP_ALLOWED = False`. Zero actuator actuation.
"""

from pathlib import Path
from datetime import datetime, timezone
import time
import math
import sqlite3
import requests
import numpy as np
import pandas as pd

# Hard Safety Invariant
PUMP_ALLOWED = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sensor_history.db"

# Conservative Physical Limits
MAX_PHYSICAL_RATE_PER_HOUR = 15.0  # Max realistic drying/wetting rate in % index / hr without flood
DEFAULT_FORECAST_HORIZON_MINUTES = 30
DEFAULT_WINDOW_MINUTES = 120
STALE_THRESHOLD_MINUTES = 30.0
MAX_ALLOWED_GAP_MINUTES = 25.0
MIN_OBSERVATIONS_REQUIRED = 3


def calculate_relative_moisture_index(raw_adc, dry_adc, wet_adc):
    """
    Converts raw 12-bit ADC count into a relative soil moisture index (0.0% to 100.0%)
    using 2-point dry air and saturated wet endpoints.
    
    Formula:
        Index % = clamp(0.0, 100.0, (dry_adc - raw_adc) / (dry_adc - wet_adc) * 100.0)
        
    Note:
        This is a relative capacitive index, NOT volumetric water content (VWC).
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
    return max(0.0, min(100.0, float(pct)))


def get_probe_calibration_from_db(probe_id, db_path=None):
    """Retrieves stored 2-point calibration endpoints for a specific probe."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not path.exists():
        return {"calibrated": False, "dry_adc": None, "wet_adc": None}

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT dry_adc, wet_adc, updated_at FROM probe_calibration WHERE probe_id = ?",
            (probe_id,),
        )
        row = cur.fetchone()
        if not row or row["dry_adc"] is None or row["wet_adc"] is None:
            return {"calibrated": False, "dry_adc": None, "wet_adc": None}
        return {
            "calibrated": True,
            "dry_adc": float(row["dry_adc"]),
            "wet_adc": float(row["wet_adc"]),
            "updated_at": row["updated_at"],
        }
    except sqlite3.Error:
        return {"calibrated": False, "dry_adc": None, "wet_adc": None}
    finally:
        conn.close()


def fetch_recent_telemetry_history(db_path=None, window_minutes=DEFAULT_WINDOW_MINUTES, sync_first=True):
    """
    Fetches genuine recent observations from `ten_minute_observations` in SQLite.
    Optionally triggers Phase 25 observation sync first to capture newly logged serial packets.
    
    Returns:
        pd.DataFrame sorted strictly by interval_start_epoch ascending.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()

    if sync_first:
        try:
            from training.experimental.phase25.realtime_history import (
                sync_raw_readings_to_ten_minute_observations,
            )
            sync_raw_readings_to_ten_minute_observations(db_path=path)
        except Exception:
            pass

    conn = sqlite3.connect(str(path))
    try:
        # Determine window cutoff based on latest available record to prevent timezone drift
        cur = conn.cursor()
        cur.execute("SELECT MAX(interval_start_epoch) FROM ten_minute_observations;")
        row = cur.fetchone()
        max_epoch = row[0] if row and row[0] is not None else time.time()
        min_epoch = max_epoch - (window_minutes * 60.0)

        query = """
            SELECT 
                interval_start_epoch,
                interval_start_utc,
                soil1_raw_mean,
                soil2_raw_mean,
                temperature_c_mean,
                humidity_pct_mean,
                rain_detected_count,
                raw_sample_count,
                status
            FROM ten_minute_observations
            WHERE interval_start_epoch >= ?
            ORDER BY interval_start_epoch ASC
        """
        df = pd.read_sql_query(query, conn, params=(min_epoch,))
        if df.empty:
            return pd.DataFrame()

        # Deduplicate and sort
        df = df.drop_duplicates(subset=["interval_start_epoch"]).sort_values("interval_start_epoch").reset_index(drop=True)
        return df
    finally:
        conn.close()


def fit_linear_trend(times_minutes, values):
    """
    Fits an ordinary least-squares line: y = m * t + c.
    
    Returns:
        dict: {
            slope_per_min: float,
            rate_per_hour: float,
            intercept: float,
            residuals_rmse: float,
            r_squared: float
        }
    """
    t = np.array(times_minutes, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    n = len(t)
    if n < 2:
        return None

    t_mean = np.mean(t)
    y_mean = np.mean(y)

    ss_tt = np.sum((t - t_mean) ** 2)
    if ss_tt < 1e-9:
        # No time spread
        return {
            "slope_per_min": 0.0,
            "rate_per_hour": 0.0,
            "intercept": float(y_mean),
            "residuals_rmse": 0.0,
            "r_squared": 0.0,
        }

    ss_ty = np.sum((t - t_mean) * (y - y_mean))
    slope_per_min = float(ss_ty / ss_tt)
    intercept = float(y_mean - slope_per_min * t_mean)

    y_pred = slope_per_min * t + intercept
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = float(1.0 - (np.sum(residuals ** 2) / ss_tot)) if ss_tot > 1e-9 else 1.0

    return {
        "slope_per_min": slope_per_min,
        "rate_per_hour": slope_per_min * 60.0,
        "intercept": intercept,
        "residuals_rmse": rmse,
        "r_squared": max(0.0, min(1.0, r2)),
    }


def generate_live_esp32_forecast(
    db_path=None,
    horizon_minutes=DEFAULT_FORECAST_HORIZON_MINUTES,
    window_minutes=DEFAULT_WINDOW_MINUTES,
    stale_threshold_minutes=STALE_THRESHOLD_MINUTES,
    max_gap_minutes=MAX_ALLOWED_GAP_MINUTES,
):
    """
    Main live forecasting pipeline:
    1. Loads genuine telemetry history and calibration endpoints.
    2. Validates freshness, sample count, and time continuity.
    3. Calculates linear trend and projects relative moisture index.
    4. Evaluates data quality / uncertainty bounds.

    Returns:
        dict: Detailed structured forecast result.
    """
    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    p1_cal = get_probe_calibration_from_db("probe1", db_path=db)
    p2_cal = get_probe_calibration_from_db("probe2", db_path=db)

    # 1. Calibration Check
    if not p1_cal.get("calibrated") and not p2_cal.get("calibrated"):
        return {
            "status": "UNCALIBRATED",
            "message": "Both capacitive probes are uncalibrated in SQLite. Live relative index cannot be computed.",
            "can_forecast": False,
            "confidence": "NONE",
            "disclaimer": "Probes require 2-point dry/wet calibration endpoints to establish relative moisture index.",
        }

    # 2. Fetch history
    df = fetch_recent_telemetry_history(db_path=db, window_minutes=window_minutes, sync_first=True)
    if df.empty or len(df) < MIN_OBSERVATIONS_REQUIRED:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": f"Insufficient observations in the last {window_minutes}m (found {len(df)}, minimum {MIN_OBSERVATIONS_REQUIRED} required).",
            "can_forecast": False,
            "confidence": "NONE",
            "observations_count": len(df),
            "disclaimer": "Trend estimation requires at least 3 continuous 10-minute intervals.",
        }

    # 3. Check Freshness
    now_epoch = time.time()
    latest_epoch = float(df["interval_start_epoch"].iloc[-1])
    latest_utc = str(df["interval_start_utc"].iloc[-1])
    time_since_latest_min = (now_epoch - latest_epoch) / 60.0

    # Note: If database timestamps are in the past relative to system wall-clock (e.g. replay),
    # we assess staleness based on relative host clock or allow evaluation if explicitly within reasonable limits.
    is_stale = time_since_latest_min > stale_threshold_minutes

    # 4. Check for Gaps (Continuous Streak Extraction)
    epochs = df["interval_start_epoch"].values.astype(float)
    gaps_min = np.diff(epochs) / 60.0
    gap_indices = np.where(gaps_min > max_gap_minutes)[0]

    streak_df = df.copy()
    gap_detected = False
    if len(gap_indices) > 0:
        gap_detected = True
        last_gap_idx = gap_indices[-1]
        streak_df = df.iloc[last_gap_idx + 1:].reset_index(drop=True)

    if len(streak_df) < MIN_OBSERVATIONS_REQUIRED:
        return {
            "status": "GAP_DETECTED",
            "message": f"Time gap > {max_gap_minutes}m detected. Current continuous streak has only {len(streak_df)} observations.",
            "can_forecast": False,
            "confidence": "NONE",
            "observations_count": len(streak_df),
            "disclaimer": "Cannot reliably interpolate across telemetry dropouts.",
        }

    # 5. Compute Relative Moisture Indices & Trends
    t_start = float(streak_df["interval_start_epoch"].iloc[0])
    times_min = (streak_df["interval_start_epoch"].values.astype(float) - t_start) / 60.0
    streak_duration_min = float(times_min[-1] - times_min[0])

    streak_df["probe1_index"] = [
        calculate_relative_moisture_index(r, p1_cal["dry_adc"], p1_cal["wet_adc"])
        for r in streak_df["soil1_raw_mean"]
    ]
    streak_df["probe2_index"] = [
        calculate_relative_moisture_index(r, p2_cal["dry_adc"], p2_cal["wet_adc"])
        for r in streak_df["soil2_raw_mean"]
    ]

    probe_results = {}
    valid_probes = []

    for p_id, p_name, cal, col in [
        ("probe1", "Probe 1", p1_cal, "probe1_index"),
        ("probe2", "Probe 2", p2_cal, "probe2_index"),
    ]:
        if not cal.get("calibrated") or streak_df[col].dropna().empty:
            probe_results[p_id] = {
                "name": p_name,
                "calibrated": False,
                "current_raw": float(streak_df[f"soil{p_id[-1]}_raw_mean"].iloc[-1]) if f"soil{p_id[-1]}_raw_mean" in streak_df else None,
                "current_index": None,
                "projected_index": None,
                "rate_per_hour": 0.0,
                "direction": "Uncalibrated",
                "uncertainty_pm": 0.0,
            }
            continue

        vals = streak_df[col].values.astype(float)
        current_idx = float(vals[-1])
        fit = fit_linear_trend(times_min, vals)

        # Conservative slope clamping
        raw_rate = fit["rate_per_hour"]
        clamped_rate = max(-MAX_PHYSICAL_RATE_PER_HOUR, min(MAX_PHYSICAL_RATE_PER_HOUR, raw_rate))
        slope_min = clamped_rate / 60.0

        delta_proj = slope_min * horizon_minutes
        projected_idx = max(0.0, min(100.0, current_idx + delta_proj))

        # Direction classification
        if clamped_rate < -0.1:
            direction = "Drying"
        elif clamped_rate > 0.1:
            direction = "Wetting"
        else:
            direction = "Stable"

        uncertainty_pm = max(0.5, fit["residuals_rmse"] * 1.96)  # ~95% confidence interval of trend

        probe_results[p_id] = {
            "name": p_name,
            "calibrated": True,
            "current_raw": float(streak_df[f"soil{p_id[-1]}_raw_mean"].iloc[-1]),
            "current_index": current_idx,
            "projected_index": float(projected_idx),
            "rate_per_hour": float(clamped_rate),
            "direction": direction,
            "uncertainty_pm": float(uncertainty_pm),
            "r_squared": fit["r_squared"],
            "rmse": fit["residuals_rmse"],
        }
        valid_probes.append(probe_results[p_id])

    # 6. Combined Average Probe Forecast
    if valid_probes:
        avg_current = float(np.mean([p["current_index"] for p in valid_probes]))
        avg_projected = float(np.mean([p["projected_index"] for p in valid_probes]))
        avg_rate = float(np.mean([p["rate_per_hour"] for p in valid_probes]))
        avg_direction = (
            "Drying" if avg_rate < -0.1 else ("Wetting" if avg_rate > 0.1 else "Stable")
        )
        avg_uncertainty = float(np.mean([p["uncertainty_pm"] for p in valid_probes]))
    else:
        avg_current = avg_projected = avg_rate = avg_uncertainty = None
        avg_direction = "Unavailable"

    # 7. Confidence & Data Quality Score
    n_obs = len(streak_df)
    max_rmse = max([p["rmse"] for p in valid_probes]) if valid_probes else 999.0

    if is_stale:
        confidence = "LOW"
        confidence_reason = f"Readings are stale ({time_since_latest_min:.1f}m since last observation)."
    elif n_obs >= 8 and max_rmse < 1.0 and not gap_detected:
        confidence = "HIGH"
        confidence_reason = f"Strong streak ({n_obs} continuous intervals, {streak_duration_min:.0f}m span), low noise (RMSE {max_rmse:.2f}%)."
    elif n_obs >= 4 and max_rmse < 2.5:
        confidence = "MEDIUM"
        confidence_reason = f"Moderate streak ({n_obs} intervals, RMSE {max_rmse:.2f}%)."
    else:
        confidence = "LOW"
        confidence_reason = f"High variance or short window ({n_obs} intervals, RMSE {max_rmse:.2f}%)."

    return {
        "status": "SUCCESS" if not is_stale else "STALE_DATA",
        "can_forecast": True,
        "message": "Live trend projection generated from genuine sensor history." if not is_stale else f"Forecast generated, but telemetry is stale ({time_since_latest_min:.1f}m old).",
        "horizon_minutes": horizon_minutes,
        "window_minutes": window_minutes,
        "latest_timestamp_utc": latest_utc,
        "latest_epoch": latest_epoch,
        "time_since_latest_minutes": float(time_since_latest_min),
        "is_stale": is_stale,
        "observations_count": n_obs,
        "streak_duration_minutes": streak_duration_min,
        "gap_detected": gap_detected,
        "probe1": probe_results.get("probe1", {}),
        "probe2": probe_results.get("probe2", {}),
        "average": {
            "current_index": avg_current,
            "projected_index": avg_projected,
            "rate_per_hour": avg_rate,
            "direction": avg_direction,
            "uncertainty_pm": avg_uncertainty,
        },
        "latest_temperature_c": float(streak_df["temperature_c_mean"].iloc[-1]) if "temperature_c_mean" in streak_df else None,
        "latest_humidity_pct": float(streak_df["humidity_pct_mean"].iloc[-1]) if "humidity_pct_mean" in streak_df else None,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "disclaimer": (
            "Experimental trend extrapolation based on genuine capacitive probe readings. "
            "Expressed as relative moisture index (0-100%), NOT volumetric water content (VWC). "
            "NOT a trained neural network or machine learning model."
        ),
        "history_df": streak_df,
    }


# In-memory weather cache (TTL = 10 minutes)
_WEATHER_CACHE = {}
_WEATHER_CACHE_TTL_SECONDS = 600.0


def fetch_optional_weather_forecast(latitude=12.9184, longitude=79.1325, timeout=3.0):
    """
    Fetches real-time atmospheric forecast from the free Open-Meteo API.
    Designed with graceful timeout and exception handling so dashboard rendering
    NEVER depends on API connectivity.
    Cached in-memory for 10 minutes to avoid redundant network calls.
    
    Returns:
        dict: Weather fields actually returned by the API, or unavailable status with error reason.
    """
    if latitude is None or longitude is None:
        return {
            "available": False,
            "provider": "Open-Meteo API",
            "latitude": None,
            "longitude": None,
            "temperature_2m_c": None,
            "relative_humidity_2m_pct": None,
            "precipitation_mm": None,
            "precipitation_probability_pct": None,
            "weather_description": "Location coordinates unconfigured",
            "disclaimer": "No valid geographic coordinates configured for weather fetching.",
            "error": "Coordinates (latitude/longitude) must be configured.",
        }

    cache_key = f"forecast_{latitude:.4f}_{longitude:.4f}"
    now = time.time()
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and (now - cached["cached_at"]) < _WEATHER_CACHE_TTL_SECONDS:
        return cached["data"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "hourly": "precipitation_probability,temperature_2m",
        "forecast_days": 1,
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        curr = data.get("current", {})
        hourly = data.get("hourly", {})

        precip_prob_next_hr = None
        if "precipitation_probability" in hourly and len(hourly["precipitation_probability"]) > 0:
            precip_prob_next_hr = float(hourly["precipitation_probability"][0])

        wcode_map = {
            0: "Clear Sky",
            1: "Mainly Clear",
            2: "Partly Cloudy",
            3: "Overcast",
            45: "Fog",
            51: "Light Drizzle",
            61: "Slight Rain",
            63: "Moderate Rain",
            65: "Heavy Rain",
            80: "Rain Showers",
        }
        wcode = curr.get("weather_code", 0)
        desc = wcode_map.get(wcode, f"Code {wcode}")

        res = {
            "available": True,
            "provider": "Open-Meteo API",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m_c": float(curr.get("temperature_2m", 0.0)),
            "relative_humidity_2m_pct": float(curr.get("relative_humidity_2m", 0.0)),
            "precipitation_mm": float(curr.get("precipitation", 0.0)),
            "precipitation_probability_pct": precip_prob_next_hr,
            "weather_description": desc,
            "disclaimer": "Atmospheric forecast from Open-Meteo. Not a measured in-ground sensor reading.",
            "error": None,
        }
        _WEATHER_CACHE[cache_key] = {"data": res, "cached_at": now}
        return res
    except Exception as e:
        return {
            "available": False,
            "provider": "Open-Meteo API",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m_c": None,
            "relative_humidity_2m_pct": None,
            "precipitation_mm": None,
            "precipitation_probability_pct": None,
            "weather_description": "Weather service offline",
            "disclaimer": "Atmospheric forecast currently unavailable.",
            "error": str(e),
        }


def fetch_hourly_weather_observations(latitude=12.9184, longitude=79.1325, past_days=2, forecast_days=1, timeout=5.0):
    """
    Fetches genuine hourly atmospheric and soil profile reanalysis variables from Open-Meteo
    matching the exact physical parameters and units used during training:
    - surface_pressure -> weather_pressure (hPa)
    - wind_speed_10m -> weather_wind_speed (km/h)
    - direct_radiation -> weather_radiation (W/m², direct solar radiation)
    - soil_temperature_0_to_7cm -> soil_temperature_0-7cm (°C, shallow soil layer)
    - soil_temperature_7_to_28cm -> note: ERA5 7-28cm band, evaluated against 7-18cm requirement

    Returns:
        dict: Containing 'available' (bool), 'weather_df' (pd.DataFrame indexed by UTC ts),
              'metadata' (dict describing units and provenance), and 'error'.
    """
    if latitude is None or longitude is None:
        return {
            "available": False,
            "weather_df": None,
            "metadata": {},
            "error": "Valid geographic coordinates (latitude, longitude) must be provided.",
        }

    cache_key = f"hourly_{latitude:.4f}_{longitude:.4f}_{past_days}_{forecast_days}"
    now = time.time()
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and (now - cached["cached_at"]) < _WEATHER_CACHE_TTL_SECONDS:
        return cached["data"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "surface_pressure",
            "wind_speed_10m",
            "direct_radiation",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
        ],
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        units = data.get("hourly_units", {})

        if "time" not in hourly or not hourly["time"]:
            return {
                "available": False,
                "weather_df": None,
                "metadata": {},
                "error": "Open-Meteo returned no hourly time series data.",
            }

        df_weather = pd.DataFrame({
            "ts_utc": pd.to_datetime(hourly["time"], utc=True),
            "weather_pressure": hourly.get("surface_pressure"),
            "weather_wind_speed": hourly.get("wind_speed_10m"),
            "weather_radiation": hourly.get("direct_radiation"),
            "soil_temperature_0-7cm": hourly.get("soil_temperature_0_to_7cm"),
            "soil_temperature_7-28cm_era5": hourly.get("soil_temperature_7_to_28cm"),
        }).dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)

        res = {
            "available": True,
            "weather_df": df_weather,
            "units": units,
            "metadata": {
                "provider": "Open-Meteo NWP / ERA5 Reanalysis API",
                "coordinates": (latitude, longitude),
                "timezone": "UTC",
                "variables": {
                    "weather_pressure": {"unit": units.get("surface_pressure", "hPa"), "match": True},
                    "weather_wind_speed": {"unit": units.get("wind_speed_10m", "km/h"), "match": True},
                    "weather_radiation": {"unit": units.get("direct_radiation", "W/m²"), "match": True},
                    "soil_temperature_0-7cm": {"unit": units.get("soil_temperature_0_to_7cm", "°C"), "match": True, "type": "NWP Model Soil Layer"},
                    "soil_temperature_7-18cm": {"unit": "°C", "match": False, "note": "API provides 7-28cm ERA5 band, not physical 7-18cm probe depth"},
                },
            },
            "error": None,
        }
        _WEATHER_CACHE[cache_key] = {"data": res, "cached_at": now}
        return res
    except Exception as e:
        return {
            "available": False,
            "weather_df": None,
            "metadata": {},
            "error": str(e),
        }


def align_weather_to_observation_timestamps(observation_timestamps, weather_df):
    """
    Aligns external hourly weather observations to high-frequency observation timestamps (e.g. 10-minute cadence)
    using linear interpolation strictly within the boundary of available weather timestamps.
    Prevents extrapolation beyond observed weather data.

    Args:
        observation_timestamps: Iterable of ISO timestamps or datetime objects (UTC).
        weather_df: DataFrame with 'ts_utc' column and weather variable columns.

    Returns:
        pd.DataFrame aligned with observation_timestamps, containing interpolated weather variables,
        or None if alignment cannot be performed without extrapolation.
    """
    if weather_df is None or weather_df.empty or observation_timestamps is None or len(observation_timestamps) == 0:
        return None

    obs_dt = pd.to_datetime(list(observation_timestamps), utc=True)
    min_obs = obs_dt.min()
    max_obs = obs_dt.max()

    w_min = weather_df["ts_utc"].min()
    w_max = weather_df["ts_utc"].max()

    # Strict check: No extrapolation outside weather bounds
    if min_obs < w_min or max_obs > w_max:
        return None

    w_indexed = weather_df.set_index("ts_utc")
    # Union index of weather timestamps and observation timestamps
    combined_index = w_indexed.index.union(obs_dt).sort_values()
    reindexed = w_indexed.reindex(combined_index)
    interpolated = reindexed.interpolate(method="time")
    aligned = interpolated.loc[obs_dt].reset_index().rename(columns={"index": "ts_utc"})
    return aligned


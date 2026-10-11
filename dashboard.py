import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import threading
import time
import sys
from pathlib import Path

from datetime import datetime, timezone

# Connect training/inference module
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    import database
except ImportError:
    database = None

try:
    from inference import (
        predict as run_model_inference,
        audit_feature_readiness,
        get_model_comparison_rows,
        get_model_trustworthiness,
        get_model_details,
        get_metrics,
    )
except ImportError:
    run_model_inference = None
    get_model_comparison_rows = None
    get_model_trustworthiness = lambda m, s=None: {
        "category": "Experimental",
        "badge": "⚪ Unknown",
        "summary": "Inference module unavailable",
        "threshold": 0.5,
        "metrics": {},
        "target": "Unknown",
        "architecture": "Unknown",
        "evaluation_type": "Unknown",
        "input_requirements": "Unknown",
        "input_readiness": "Unavailable",
        "input_readiness_code": "UNAVAILABLE",
    }
    get_model_details = None
    get_metrics = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

ACTIVE_MODEL_NAME = "Random Forest (11-Feature)"
DEFAULT_MODEL = ACTIVE_MODEL_NAME

DEFAULT_STATE = {
    "uploaded_image": None,
    "weed_detected": None,
    "weed_name": None,
    "weed_confidence": None,
    "recommendation": None,
    "last_detection_time": None,
    "selected_model": ACTIVE_MODEL_NAME,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.session_state["selected_model"] = ACTIVE_MODEL_NAME


# ============================================================
# SETTINGS
# ============================================================

# ESP32 USB serial connection. Close Arduino Serial Monitor before running Streamlit.
ESP32_SERIAL_PORT = "/dev/cu.usbserial-0001"
ESP32_BAUD_RATE = 115200
ESP32_STALE_AFTER_SECONDS = 25

# Demo values remain disabled: only real sensor readings are shown.
USE_DEMO_DATA = False

# ThingSpeak settings
THINGSPEAK_CHANNEL_ID = ""
THINGSPEAK_READ_API_KEY = ""

# YOLO model path
YOLO_MODEL_PATH = "models/weed_model.pt"


# ============================================================
# WEATHER LOCATION
# ============================================================

# Display name
WEATHER_CITY = "Vellore"

# Vellore city coordinates
WEATHER_LATITUDE = 12.9184
WEATHER_LONGITUDE = 79.1325

# Weather provider
WEATHER_PROVIDER = "Open-Meteo"


# ============================================================
# HELPER
# ============================================================

def display_value(value, suffix=""):
    """
    Display a value if it exists.
    Otherwise return N/A.
    """

    if value is None:
        return "N/A"

    return f"{value}{suffix}"


# ============================================================
# SENSOR DATA
# ============================================================

def get_demo_sensor_data():
    """
    Demo sensor data is intentionally disabled.

    We return N/A instead of fake values.
    """

    return {
        "temperature": None,
        "humidity": None,
        "soil_moisture": None,
        "water_level": None,
        "rain": None,
        "pump_status": None,
    }


def get_thingspeak_data(channel_id, api_key):
    """
    Read sensor data from ThingSpeak.

    Expected fields:

    field1 = soil moisture
    field2 = temperature
    field3 = humidity
    field4 = water level
    field5 = rain
    field6 = pump status
    """

    if not channel_id or not api_key:
        return None

    url = (
        f"https://api.thingspeak.com/"
        f"channels/{channel_id}/feeds.json"
    )

    try:
        response = requests.get(
            url,
            params={
                "api_key": api_key,
                "results": 10,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()
        feeds = data.get("feeds", [])

        if not feeds:
            return None

        latest = feeds[-1]

        def to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        rain_value = latest.get("field5")

        if rain_value is None:
            rain = None
        else:
            rain = str(rain_value).lower() in [
                "1",
                "true",
                "yes",
            ]

        return {
            "temperature": to_float(
                latest.get("field2")
            ),
            "humidity": to_float(
                latest.get("field3")
            ),
            "soil_moisture": to_float(
                latest.get("field1")
            ),
            "water_level": to_float(
                latest.get("field4")
            ),
            "rain": rain,
            "pump_status": latest.get("field6"),
        }

    except requests.RequestException:
        return None


class ESP32SerialReader:
    """Keep one serial connection alive while Streamlit reruns the script."""

    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.latest = None
        self.last_received_at = None
        self.error = None
        self.lock = threading.Lock()
        self.serial_connection = None
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            import serial
        except ImportError:
            with self.lock:
                self.error = "pyserial is not installed. Run: python3 -m pip install pyserial"
            return

        while True:
            try:
                if self.serial_connection is None or not self.serial_connection.is_open:
                    self.serial_connection = serial.Serial(
                        self.port,
                        self.baud_rate,
                        timeout=2,
                    )
                    # Match the successful USB test setup for this ESP32.
                    self.serial_connection.dtr = False
                    self.serial_connection.rts = False
                    time.sleep(2)
                    self.serial_connection.reset_input_buffer()

                line = self.serial_connection.readline().decode(
                    "utf-8", errors="replace"
                ).strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore startup messages or incomplete/non-JSON serial lines.
                    continue

                if not isinstance(payload, dict):
                    continue

                # Safely log genuine telemetry to persistent SQLite database
                if database is not None:
                    try:
                        database.log_reading(payload, source="esp32_usb", status="valid")
                    except Exception:
                        pass

                with self.lock:
                    self.latest = payload
                    self.last_received_at = time.time()
                    self.error = None

            except (OSError, Exception) as exc:
                # Includes serial port busy/disconnected and pyserial exceptions.
                with self.lock:
                    self.error = f"Cannot read ESP32 on {self.port}: {exc}"
                if self.serial_connection is not None:
                    try:
                        self.serial_connection.close()
                    except Exception:
                        pass
                    self.serial_connection = None
                time.sleep(2)

    def get_data(self):
        with self.lock:
            data = dict(self.latest) if self.latest is not None else None
            received_at = self.last_received_at
            error = self.error

        if data is not None and received_at is not None:
            if time.time() - received_at > ESP32_STALE_AFTER_SECONDS:
                return None, "ESP32 data is stale. Check the board, USB cable, and serial port."
        return data, error


@st.cache_resource
def get_esp32_reader():
    return ESP32SerialReader(ESP32_SERIAL_PORT, ESP32_BAUD_RATE)


def get_sensor_data():
    """Read live sensor values from the ESP32 USB serial JSON stream."""
    reader = get_esp32_reader()
    raw, error = reader.get_data()

    calibrations = database.get_all_calibrations() if database is not None else {}
    cal1 = calibrations.get("probe1", {})
    cal2 = calibrations.get("probe2", {})

    if raw is None:
        return {
            "temperature": None,
            "humidity": None,
            "soil_moisture": None,
            "soil_moisture_calibrated": None,
            "is_calibrated": False,
            "soil1_raw": None,
            "soil2_raw": None,
            "soil1_calibrated": None,
            "soil2_calibrated": None,
            "probe1_calibrated": cal1.get("calibrated", False),
            "probe2_calibrated": cal2.get("calibrated", False),
            "probe1_calibration": cal1,
            "probe2_calibration": cal2,
            "water_level": None,
            "rain": None,
            "pump_status": None,
            "sensor_error": error or "Waiting for the first ESP32 reading...",
            "last_received_at": None,
        }

    def numeric_value(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    soil1 = numeric_value(raw.get("soil1_raw"))
    soil2 = numeric_value(raw.get("soil2_raw"))
    soil_values = [value for value in (soil1, soil2) if value is not None]
    soil_average = sum(soil_values) / len(soil_values) if soil_values else None

    # Calculate calibrated soil moisture using independent probe calibration references
    probe1_cal = cal1.get("calibrated", False)
    probe2_cal = cal2.get("calibrated", False)

    soil1_calibrated = None
    if probe1_cal and soil1 is not None and database is not None:
        soil1_calibrated = database.calculate_calibrated_moisture(soil1, cal1["dry_adc"], cal1["wet_adc"])

    soil2_calibrated = None
    if probe2_cal and soil2 is not None and database is not None:
        soil2_calibrated = database.calculate_calibrated_moisture(soil2, cal2["dry_adc"], cal2["wet_adc"])

    calibrated_vals = [v for v in (soil1_calibrated, soil2_calibrated) if v is not None]
    soil_moisture_calibrated = sum(calibrated_vals) / len(calibrated_vals) if calibrated_vals else None
    is_calibrated = (soil_moisture_calibrated is not None)

    rain_value = raw.get("rain_detected")
    if isinstance(rain_value, bool):
        rain = rain_value
    elif isinstance(rain_value, (int, float)):
        rain = bool(rain_value)
    elif isinstance(rain_value, str) and rain_value.strip().lower() in {
        "true", "1", "yes", "rain"
    }:
        rain = True
    elif isinstance(rain_value, str) and rain_value.strip().lower() in {
        "false", "0", "no", "dry"
    }:
        rain = False
    else:
        rain = None

    return {
        "temperature": numeric_value(raw.get("temperature_c")),
        "humidity": numeric_value(raw.get("humidity_pct")),
        "soil_moisture": soil_average,
        "soil_moisture_calibrated": soil_moisture_calibrated,
        "is_calibrated": is_calibrated,
        "soil1_raw": soil1,
        "soil2_raw": soil2,
        "soil1_calibrated": soil1_calibrated,
        "soil2_calibrated": soil2_calibrated,
        "probe1_calibrated": probe1_cal,
        "probe2_calibrated": probe2_cal,
        "probe1_calibration": cal1,
        "probe2_calibration": cal2,
        "water_level": numeric_value(raw.get("water_level_raw")),
        "rain": rain,
        "pump_status": None,
        "sensor_error": None,
        "last_received_at": datetime.now().strftime("%H:%M:%S"),
    }


# ============================================================
# WEATHER - OPEN-METEO
# ============================================================

@st.cache_data(ttl=600)
def get_weather_data(latitude, longitude):
    """
    Get current weather + hourly forecast
    from Open-Meteo.

    Cached for 10 minutes.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        # Current conditions
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "surface_pressure,"
            "weather_code"
        ),

        # Hourly forecast
        "hourly": (
            "temperature_2m,"
            "precipitation_probability,"
            "precipitation,"
            "weather_code"
        ),

        # Use the location's local timezone
        "timezone": "auto",

        # Two days gives us enough data
        # for a complete 24-hour forecast.
        "forecast_days": 2,

        # Units
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # WEATHER CODE MAPPING
        # ----------------------------------------------------

        weather_code_names = {
            0: "Clear Sky",
            1: "Mainly Clear",
            2: "Partly Cloudy",
            3: "Overcast Clouds",

            45: "Fog",
            48: "Depositing Rime Fog",

            51: "Light Drizzle",
            53: "Moderate Drizzle",
            55: "Dense Drizzle",

            56: "Light Freezing Drizzle",
            57: "Dense Freezing Drizzle",

            61: "Slight Rain",
            63: "Moderate Rain",
            65: "Heavy Rain",

            66: "Light Freezing Rain",
            67: "Heavy Freezing Rain",

            71: "Slight Snow",
            73: "Moderate Snow",
            75: "Heavy Snow",

            77: "Snow Grains",

            80: "Slight Rain Showers",
            81: "Moderate Rain Showers",
            82: "Violent Rain Showers",

            85: "Slight Snow Showers",
            86: "Heavy Snow Showers",

            95: "Thunderstorm",
            96: "Thunderstorm with Slight Hail",
            99: "Thunderstorm with Heavy Hail",
        }

        current = data["current"]
        hourly = data["hourly"]

        # ----------------------------------------------------
        # CURRENT WEATHER
        # ----------------------------------------------------

        current_code = current["weather_code"]

        current_weather = {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_gust": current.get("wind_gusts_10m"),
            "pressure": current.get("surface_pressure"),
            "description": weather_code_names.get(
                current_code,
                "Unknown",
            ),
            "fetched_at": datetime.now(timezone.utc),
        }

        # ----------------------------------------------------
        # HOURLY FORECAST
        # ----------------------------------------------------

        forecast = []

        current_local_time = datetime.fromisoformat(
            current["time"]
        )

        for i, time_string in enumerate(
            hourly["time"]
        ):

            forecast_time = datetime.fromisoformat(
                time_string
            )

            # Ignore hours already passed.
            if forecast_time < current_local_time:
                continue

            probability = hourly[
                "precipitation_probability"
            ][i]

            precipitation = hourly[
                "precipitation"
            ][i]

            weather_code = hourly[
                "weather_code"
            ][i]

            forecast.append(
                {
                    "time": forecast_time,
                    "temperature": hourly[
                        "temperature_2m"
                    ][i],
                    "rain_probability": probability,
                    "rain_amount": precipitation,
                    "description": weather_code_names.get(
                        weather_code,
                        "Unknown",
                    ),
                }
            )

            # We only need the next 24 hourly periods.
            if len(forecast) >= 24:
                break

        return {
            "current": current_weather,
            "forecast": forecast,
        }

    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# RAIN SUMMARY
# ============================================================

def get_rain_summary(forecast_data):
    """
    Find the HIGHEST hourly precipitation probability
    inside each time window.

    These are peak probabilities, not a single probability
    that rain will occur continuously throughout the window.
    """

    if not forecast_data:
        return {
            "6h": None,
            "12h": None,
            "24h": None,
        }

    return {
        "6h": max(
            item["rain_probability"]
            for item in forecast_data[:6]
        ),
        "12h": max(
            item["rain_probability"]
            for item in forecast_data[:12]
        ),
        "24h": max(
            item["rain_probability"]
            for item in forecast_data[:24]
        ),
    }


# ============================================================
# IRRIGATION AI
# ============================================================

def predict_irrigation(
    sensor_data,
    weather_data,
    model_name=DEFAULT_MODEL,
    enforce_complete_readiness=False,
):
    """
    Executes irrigation prediction using the active 11-feature model.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    if run_model_inference is None:
        return {
            "model": model_name,
            "required": None,
            "message": "Inference module is not available.",
            "confidence": None,
            "threshold": 0.5,
        }

    try:
        result = run_model_inference(
            model_name=model_name,
            sensor_data=sensor_data,
            weather_data=weather_data,
            enforce_complete_readiness=enforce_complete_readiness,
        )
        return {
            "model": model_name,
            "required": result.get("required"),
            "message": result.get("message", "N/A"),
            "confidence": result.get("confidence"),
            "threshold": result.get("threshold"),
            "readiness": result.get("readiness"),
        }
    except Exception as exc:
        return {
            "model": model_name,
            "required": None,
            "message": f"Inference error: {exc}",
            "confidence": None,
            "threshold": 0.5,
        }


# ============================================================
# WEED DETECTION
# ============================================================

def run_weed_detection(image_file):
    """
    YOLO model is not integrated yet.

    Therefore we return N/A.
    """

    return {
        "weed_detected": None,
        "weed_name": None,
        "confidence": None,
    }


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

def get_treatment_recommendation(
    weed_name,
    crop_name,
    weather_data,
):
    """
    Treatment recommendation system is not integrated yet.

    Therefore we return N/A.
    """

    return {
        "weed_name": weed_name,
        "crop": crop_name,
        "treatment": None,
        "dose": None,
        "spray_window": None,
        "duration": None,
        "weather_status": None,
    }


# ============================================================
# SENSOR HISTORY
# ============================================================

def get_sensor_history():
    """
    Safely retrieves genuine sensor history from persistent SQLite database.
    Does not substitute synthetic readings.
    """
    if database is None:
        return None

    try:
        df = database.get_history(limit_hours=24)
        if df.empty:
            return None

        calibrations = database.get_all_calibrations()
        cal1 = calibrations.get("probe1", {})
        cal2 = calibrations.get("probe2", {})
        probe1_cal = cal1.get("calibrated", False)
        probe2_cal = cal2.get("calibrated", False)

        # Format Time for charts
        df["Time"] = pd.to_datetime(df["recorded_at"], unit="s").dt.strftime("%H:%M:%S")

        # Compute calibrated moisture if calibration exists, else raw ADC average
        def calc_row_moisture(row):
            vals = []
            if probe1_cal and pd.notna(row.get("soil1_raw")):
                m1 = database.calculate_calibrated_moisture(row["soil1_raw"], cal1["dry_adc"], cal1["wet_adc"])
                if m1 is not None:
                    vals.append(m1)
            if probe2_cal and pd.notna(row.get("soil2_raw")):
                m2 = database.calculate_calibrated_moisture(row["soil2_raw"], cal2["dry_adc"], cal2["wet_adc"])
                if m2 is not None:
                    vals.append(m2)
            if vals:
                return round(sum(vals) / len(vals), 1)
            # Fall back to raw ADC average
            raw_vals = [r for r in (row.get("soil1_raw"), row.get("soil2_raw")) if pd.notna(r)]
            return round(sum(raw_vals) / len(raw_vals), 0) if raw_vals else None

        df["Soil Moisture"] = df.apply(calc_row_moisture, axis=1)
        return df
    except Exception:
        return None


# ============================================================
# WATER HISTORY
# ============================================================

def get_water_history():
    """
    Historical water usage data is not connected yet.
    """

    return None


# ============================================================
# LOAD DATA
# ============================================================

sensor_data = get_sensor_data()

weather_result = get_weather_data(
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
)

if weather_result is not None:

    weather_data = weather_result["current"]
    forecast_data = weather_result["forecast"]

else:

    weather_data = {
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "wind_gust": None,
        "pressure": None,
        "description": "N/A",
        "fetched_at": None,
    }

    forecast_data = []


rain_summary = get_rain_summary(
    forecast_data
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌱 Smart Agriculture"
)

st.sidebar.caption(
    "Precision Agriculture Dashboard"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "💧 Irrigation",
        "🌿 Weed Detection",
        "🌦 Weather",
        "🧪 Recommendations",
        "🤖 AI Models",
    ],
    key="navigation_radio",
)

st.sidebar.divider()

st.sidebar.subheader(
    "⚙️ Settings"
)

crop_name = st.sidebar.selectbox(
    "Crop",
    [
        "Select crop",
        "Wheat",
        "Rice",
        "Maize",
        "Cotton",
        "Vegetables",
        "Other",
    ],
)

model_name = ACTIVE_MODEL_NAME
st.session_state["selected_model"] = model_name
trust_info = get_model_trustworthiness(model_name, sensor_data)
st.sidebar.markdown(
    f"<div style='margin-top: 6px; margin-bottom: 12px; padding: 8px 10px; border-radius: 6px; background-color: rgba(245, 127, 23, 0.08); border-left: 3px solid #f57f17; font-size: 0.82rem;'>"
    f"<b>Active Model:</b> {model_name}<br/>"
    f"<b>Status:</b> <span style='color: #f57f17; font-weight: 600;'>Experimental — Advisory Only</span><br/>"
    f"<span style='color: #555; font-size: 0.77rem;'>HistGradientBoosting (11 features) • Tuned Threshold: <code>{trust_info['threshold']:.3f}</code></span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SOIL PROBE CALIBRATION INTERFACE (TASK 2)
# ------------------------------------------------------------
cal_expand = st.sidebar.expander("🌱 Soil Probe Calibration", expanded=False)
with cal_expand:
    st.markdown("##### Probe Calibration (Estimated Relative %)")
    st.caption(
        "Enter measured dry (in air) and wet (submerged in water) ADC reference values. "
        "Each probe is calibrated independently. Values represent an estimated relative moisture "
        "percentage (0–100%), not physical volumetric water content (VWC)."
    )
    probe_choice = st.radio(
        "Select Probe",
        ["Probe 1 (GPIO 32)", "Probe 2 (GPIO 33)"],
        horizontal=True,
        key="cal_probe_choice",
    )
    p_id = "probe1" if "Probe 1" in probe_choice else "probe2"
    p_raw = sensor_data.get("soil1_raw" if p_id == "probe1" else "soil2_raw")

    if p_raw is not None:
        st.info(f"📡 Current Live Reading: **{p_raw:g} ADC**")
    else:
        st.caption("📡 Current Live Reading: N/A (ESP32 disconnected)")

    p_cal = sensor_data.get(f"{p_id}_calibration", {}) if database else {}
    if not p_cal and database:
        p_cal = database.get_probe_calibration(p_id)

    if p_cal.get("calibrated"):
        st.success(
            f"🟢 **Calibrated (Provisional Relative %)**\n"
            f"- Dry Reference: `{p_cal['dry_adc']:.0f}`\n"
            f"- Wet Reference: `{p_cal['wet_adc']:.0f}`\n"
            f"- Updated: `{str(p_cal.get('updated_at', 'N/A'))[:19]}`"
        )
    else:
        st.warning("🔴 **Uncalibrated** (Raw ADC only)")

    # Initial provisional calibration values:
    # Sensor 1 (GPIO 32): dry = 2054, wet = 1073
    # Sensor 2 (GPIO 33): dry = 1906, wet = 1023
    prov_dry = 2054 if p_id == "probe1" else 1906
    prov_wet = 1073 if p_id == "probe1" else 1023

    default_dry = int(p_cal.get("dry_adc", prov_dry)) if p_cal.get("calibrated") else prov_dry
    default_wet = int(p_cal.get("wet_adc", prov_wet)) if p_cal.get("calibrated") else prov_wet

    input_dry = st.number_input(
        f"{probe_choice} Dry ADC (in air)",
        min_value=0,
        max_value=4095,
        value=default_dry,
        step=10,
        key=f"input_dry_{p_id}",
        help="Higher ADC count when sensor is dry in air."
    )
    input_wet = st.number_input(
        f"{probe_choice} Wet ADC (in water)",
        min_value=0,
        max_value=4095,
        value=default_wet,
        step=10,
        key=f"input_wet_{p_id}",
        help="Lower ADC count when sensor is submerged in water."
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("💾 Save", key=f"save_btn_{p_id}", use_container_width=True):
            if database is not None:
                ok, err = database.save_probe_calibration(p_id, input_dry, input_wet)
                if ok:
                    st.success(f"Saved calibration for {probe_choice}!")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")
    with btn_col2:
        if st.button("🗑 Reset", key=f"reset_btn_{p_id}", use_container_width=True):
            if database is not None:
                try:
                    conn = database.get_connection()
                    with conn:
                        conn.execute("DELETE FROM probe_calibration WHERE probe_id = ?;", (p_id,))
                    conn.close()
                    st.info(f"Reset {probe_choice} to uncalibrated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reset error: {exc}")

st.sidebar.divider()

# ============================================================
# WEATHER REFRESH
# ============================================================

if st.sidebar.button(
    "🔄 Refresh Weather",
    use_container_width=True,
):

    get_weather_data.clear()

    st.rerun()


if sensor_data.get("sensor_error"):
    st.sidebar.warning(f"ESP32: {sensor_data['sensor_error']}")
else:
    st.sidebar.success(
        "ESP32 connected • Last reading "
        f"{sensor_data.get('last_received_at', 'N/A')}"
    )


# Run irrigation prediction using selected model and current data (safeguarding live predictions)
irrigation_prediction = predict_irrigation(
    sensor_data=sensor_data,
    weather_data=weather_data,
    model_name=model_name,
    enforce_complete_readiness=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title(
        "🌱 Smart Agriculture Dashboard"
    )

    st.write(
        "Integrated monitoring for smart irrigation, "
        "weed detection, weather-aware decisions, "
        "and AI-based agricultural recommendations."
    )

    st.divider()

    st.header(
        "📊 Current Field Status",
        anchor=False,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌡 Temperature",
        display_value(
            sensor_data["temperature"],
            " °C",
        ),
    )

    if sensor_data.get("is_calibrated"):
        c2.metric(
            "🌱 Soil Moisture",
            f"{sensor_data['soil_moisture_calibrated']:.1f} %",
        )
    else:
        c2.metric(
            "🌱 Soil Sensor Average (raw ADC)",
            display_value(sensor_data["soil_moisture"]),
        )

    c3.metric(
        "💦 Humidity",
        display_value(
            sensor_data["humidity"],
            " %",
        ),
    )

    c4.metric(
        "🪣 Water Sensor (raw ADC)",
        display_value(sensor_data["water_level"]),
    )

    cal1_info = sensor_data.get("probe1_calibration", {})
    cal2_info = sensor_data.get("probe2_calibration", {})
    probe1_cal = cal1_info.get("calibrated", False)
    probe2_cal = cal2_info.get("calibrated", False)

    probe1, probe2 = st.columns(2)
    if probe1_cal and sensor_data.get("soil1_calibrated") is not None:
        probe1.metric(
            "Soil Sensor 1 (Relative %)",
            f"{sensor_data['soil1_calibrated']:.1f} %",
            f"Raw: {sensor_data.get('soil1_raw')} ADC",
        )
    else:
        probe1.metric(
            "Soil Sensor 1 (raw ADC)",
            display_value(sensor_data.get("soil1_raw")),
            "🔴 Uncalibrated",
        )

    if probe2_cal and sensor_data.get("soil2_calibrated") is not None:
        probe2.metric(
            "Soil Sensor 2 (Relative %)",
            f"{sensor_data['soil2_calibrated']:.1f} %",
            f"Raw: {sensor_data.get('soil2_raw')} ADC",
        )
    else:
        probe2.metric(
            "Soil Sensor 2 (raw ADC)",
            display_value(sensor_data.get("soil2_raw")),
            "🔴 Uncalibrated",
        )

    if not (probe1_cal or probe2_cal):
        st.caption(
            "Soil readings are raw analog-to-digital converter (ADC) counts (0–4095). "
            "Use '🌱 Soil Probe Calibration' in the sidebar to configure measured dry and wet reference values. "
            "Percentages represent estimated relative moisture (0–100%), not physical volumetric water content (VWC)."
        )
    else:
        st.caption(
            f"Soil probes: Probe 1 ({'🟢 Calibrated (Relative %)' if probe1_cal else '🔴 Uncalibrated'}), "
            f"Probe 2 ({'🟢 Calibrated (Relative %)' if probe2_cal else '🔴 Uncalibrated'}). "
            "Calibrated percentages indicate estimated relative moisture index, not physical volumetric water content (VWC). "
            "Water level is hardware telemetry (raw ADC)."
        )

    st.divider()

    st.header(
        "⚡ System Status",
        anchor=False,
    )

    s1, s2, s3 = st.columns(3)

    if sensor_data["rain"] is None:
        s1.info("🌧 Rain Detection: N/A")
    elif sensor_data["rain"]:
        s1.warning("🌧 Rain Detected")
    else:
        s1.success("☀️ No Rain Detected")

    if sensor_data["water_level"] is None:
        s2.info("🪣 Water Sensor: N/A")
    else:
        s2.info(
            "🪣 Water Sensor Raw ADC: "
            f"{sensor_data['water_level']:g}"
        )

    if sensor_data["pump_status"] is None:
        s3.info("🔌 Pump Status: N/A")
    else:
        s3.info(
            f"🔌 Pump: {sensor_data['pump_status']}"
        )

    st.divider()

    st.header(
        "💧 Irrigation Prediction",
        anchor=False,
    )

    om1, om2, om3 = st.columns(3)
    om1.metric("Selected Model", model_name)
    if irrigation_prediction["required"] is None:
        ov_decision = "N/A"
    elif irrigation_prediction["required"]:
        ov_decision = "🚨 Irrigate Needed"
    else:
        ov_decision = "✅ No Irrigation"
    om2.metric("Irrigation Decision", ov_decision)
    thresh_val = irrigation_prediction.get("threshold", trust_info.get("threshold", 0.5))
    om3.metric("Decision Threshold", f"{thresh_val:.3f}")

    ov_badge_color = "#2e7d32" if trust_info["category"] == "Best-supported" else "#c62828"
    st.markdown(
        f"<div style='margin-bottom: 12px; padding: 6px 12px; border-radius: 6px; background-color: rgba(128,128,128,0.07); border-left: 3px solid {ov_badge_color}; font-size: 0.85rem;'>"
        f"<b>Selected Model:</b> {model_name} &nbsp;|&nbsp; <b>Trustworthiness:</b> <span style='color: {ov_badge_color}; font-weight: 600;'>{trust_info['category']}</span> &nbsp;—&nbsp; <i>{trust_info['evaluation_type']}</i>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if irrigation_prediction["required"] is None:
        st.warning(
            f"⚠️ **Irrigation AI (N/A):** {irrigation_prediction['message']}"
        )
    elif irrigation_prediction["required"]:
        conf_str = f" (Confidence: {irrigation_prediction['confidence'] * 100:.1f}%)" if irrigation_prediction["confidence"] is not None else ""
        st.error(
            f"🚨 **Irrigation Needed**{conf_str} — {irrigation_prediction['message']}"
        )
    else:
        conf_str = f" (Confidence: {(1 - irrigation_prediction['confidence']) * 100:.1f}%)" if irrigation_prediction["confidence"] is not None else ""
        st.success(
            f"✅ **No Irrigation Needed**{conf_str} — {irrigation_prediction['message']}"
        )

    st.divider()

    st.header(
        "🌦 Current Weather",
        anchor=False,
    )

    w1, w2, w3 = st.columns(3)

    w1.metric(
        "Temperature",
        display_value(
            weather_data["temperature"],
            " °C",
        ),
    )

    w2.metric(
        "Humidity",
        display_value(
            weather_data["humidity"],
            " %",
        ),
    )

    w3.metric(
        "Wind Speed",
        display_value(
            weather_data["wind_speed"],
            " m/s",
        ),
    )

    st.write(
        "Condition: "
        f"{weather_data['description']}"
    )

    st.divider()

    st.header(
        "🌧 Forecast Summary",
        anchor=False,
    )

    rs1, rs2, rs3 = st.columns(3)

    rs1.metric(
        "Peak Rain Probability — Next 6h",
        display_value(
            rain_summary["6h"],
            "%",
        ),
    )

    rs2.metric(
        "Peak Rain Probability — Next 12h",
        display_value(
            rain_summary["12h"],
            "%",
        ),
    )

    rs3.metric(
        "Peak Rain Probability — Next 24h",
        display_value(
            rain_summary["24h"],
            "%",
        ),
    )


# ============================================================
# IRRIGATION
# ============================================================

elif page == "💧 Irrigation":

    st.title(
        "💧 Smart Irrigation"
    )

    st.write(
        "Sensor monitoring + AI prediction + "
        "weather-aware scheduling."
    )

    st.divider()

    st.header(
        "Current Sensor Readings",
        anchor=False,
    )

    c1, c2, c3, c4 = st.columns(4)

    if sensor_data.get("is_calibrated"):
        c1.metric(
            "Soil Moisture",
            f"{sensor_data['soil_moisture_calibrated']:.1f} %",
        )
    else:
        c1.metric(
            "Soil Sensor Average (raw ADC)",
            display_value(sensor_data["soil_moisture"]),
        )

    c2.metric(
        "Temperature",
        display_value(
            sensor_data["temperature"],
            " °C",
        ),
    )

    c3.metric(
        "Humidity",
        display_value(
            sensor_data["humidity"],
            " %",
        ),
    )

    c4.metric(
        "Water Sensor (raw ADC)",
        display_value(sensor_data["water_level"]),
    )

    cal1_info = sensor_data.get("probe1_calibration", {})
    cal2_info = sensor_data.get("probe2_calibration", {})
    probe1_cal = cal1_info.get("calibrated", False)
    probe2_cal = cal2_info.get("calibrated", False)

    probe1, probe2 = st.columns(2)
    if probe1_cal and sensor_data.get("soil1_calibrated") is not None:
        probe1.metric(
            "Soil Sensor 1 (Relative %)",
            f"{sensor_data['soil1_calibrated']:.1f} %",
            f"Raw: {sensor_data.get('soil1_raw')} ADC",
        )
    else:
        probe1.metric(
            "Soil Sensor 1 (raw ADC)",
            display_value(sensor_data.get("soil1_raw")),
            "🔴 Uncalibrated",
        )

    if probe2_cal and sensor_data.get("soil2_calibrated") is not None:
        probe2.metric(
            "Soil Sensor 2 (Relative %)",
            f"{sensor_data['soil2_calibrated']:.1f} %",
            f"Raw: {sensor_data.get('soil2_raw')} ADC",
        )
    else:
        probe2.metric(
            "Soil Sensor 2 (raw ADC)",
            display_value(sensor_data.get("soil2_raw")),
            "🔴 Uncalibrated",
        )

    st.caption(
        "Soil probes are independently calibrated via SQLite references to estimated relative moisture percentage (0–100%), "
        "not physical volumetric water content (VWC). Water sensor is hardware telemetry (raw ADC)."
    )

    st.divider()

    st.header(
        "🧠 AI Irrigation Prediction",
        anchor=False,
    )

    model_name = ACTIVE_MODEL_NAME
    st.session_state["selected_model"] = model_name
    trust_info = get_model_trustworthiness(model_name, sensor_data)
    irrigation_prediction = predict_irrigation(
        sensor_data=sensor_data,
        weather_data=weather_data,
        model_name=model_name,
        enforce_complete_readiness=True,
    )

    st.markdown(
        f"<div style='margin-bottom: 16px; padding: 10px 14px; border-radius: 6px; background-color: rgba(245, 127, 23, 0.08); border-left: 4px solid #f57f17; font-size: 0.88rem;'>"
        f"<b>Active Model:</b> {model_name} &nbsp;|&nbsp; <b>Status:</b> <span style='color: #f57f17; font-weight: 600;'>Experimental — Advisory Only</span><br/>"
        f"<span style='font-size: 0.82rem; color: #555;'>Sole active prediction model • 11 environmental features • Tuned Decision Threshold: <code>{trust_info['threshold']:.4f}</code></span><br/>"
        f"<span style='font-size: 0.80rem; color: #b71c1c; font-weight: 500;'>⚠️ Warning: Relative soil moisture from capacitive probes has not been field-validated against dataset scale. Predictions are strictly advisory; automatic pump actuation is disabled.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Key metrics columns: Decision, Confidence, Decision Threshold, Pump Status
    p1, p2, p3, p4 = st.columns(4)

    if irrigation_prediction["required"] is None:
        decision_label = "N/A"
    elif irrigation_prediction["required"]:
        decision_label = "🚨 Irrigate Needed"
    else:
        decision_label = "✅ No Irrigation"

    p1.metric(
        "Irrigation Decision",
        decision_label,
    )

    if irrigation_prediction.get("confidence") is not None:
        p2.metric(
            "Model Confidence",
            f"{irrigation_prediction['confidence'] * 100:.1f}%",
        )
    else:
        p2.metric(
            "Model Confidence",
            "N/A",
        )

    thresh_val = irrigation_prediction.get("threshold", trust_info.get("threshold", 0.5))
    p3.metric(
        "Decision Threshold",
        f"{thresh_val:.3f}",
    )

    p4.metric(
        "Pump Status",
        display_value(
            sensor_data["pump_status"]
        ),
    )

    # Status / Explanation Alert
    if irrigation_prediction["required"] is None:
        st.warning(
            f"⚠️ **Prediction Status: N/A** — {irrigation_prediction['message']}"
        )
    elif irrigation_prediction["required"]:
        st.error(
            f"🚨 **Decision: Irrigation Needed** (Confidence: {irrigation_prediction['confidence'] * 100:.1f}%) — {irrigation_prediction['message']}"
        )
    else:
        st.success(
            f"✅ **Decision: No Irrigation Needed** (Confidence: {(1 - irrigation_prediction['confidence']) * 100:.1f}%) — {irrigation_prediction['message']}"
        )

    # Two Structured Sub-cards: Model Evaluation Reliability vs Current Input Readiness
    col_rel, col_ready = st.columns(2)
    with col_rel:
        st.markdown("##### 🔬 Model Evaluation Reliability")
        st.markdown(f"- **Target Task:** {trust_info['target']}")
        st.markdown(f"- **Evaluation Dataset:** {trust_info['evaluation_type']}")
        st.markdown(f"- **Architecture:** {trust_info['architecture']}")

        # Display key metrics
        if trust_info.get("metrics"):
            m_items = list(trust_info["metrics"].items())
            m_cols = st.columns(min(len(m_items), 3))
            for i, (k, v) in enumerate(m_items[:6]):
                m_cols[i % min(len(m_items), 3)].metric(k, v)

        st.caption(f"**Verification Verdict:** {trust_info['summary']}")

    with col_ready:
        st.markdown("##### 🔌 Current Input Readiness")
        st.markdown(f"- **Readiness State:** `{trust_info['input_readiness']}`")
        st.markdown(f"- **Expected Pipeline Inputs:** {trust_info['input_requirements']}")
        audit_rf = audit_feature_readiness(sensor_data=sensor_data, weather_data=weather_data, model_name=model_name)
        st.markdown(f"- **Observed Inputs Ready:** `{audit_rf['ready_count']} / {audit_rf['total_features']}`")
        if audit_rf["is_live_ready"]:
            st.success("✅ Complete 11-feature set ready for live prediction (Advisory Only).")
        else:
            st.error(f"🛑 Live prediction withheld as N/A: {len(audit_rf['blockers'])} feature blocker(s) active.")
            with st.expander("🔍 View 11-Feature Provenance & Readiness Table", expanded=False):
                audit_df = pd.DataFrame(audit_rf["feature_rows"])[
                    ["Feature", "Units", "Live Value", "Source", "Provenance", "Ready", "Limitation / Details"]
                ]
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
                st.caption("Training medians are not silently substituted for unobserved sensors in live inference.")

    st.divider()

    st.header(
        "🌦 Weather-Aware Scheduling",
        anchor=False,
    )

    rs1, rs2, rs3 = st.columns(3)

    rs1.metric(
        "Peak Rain Probability — Next 6h",
        display_value(
            rain_summary["6h"],
            "%",
        ),
    )

    rs2.metric(
        "Peak Rain Probability — Next 12h",
        display_value(
            rain_summary["12h"],
            "%",
        ),
    )

    rs3.metric(
        "Peak Rain Probability — Next 24h",
        display_value(
            rain_summary["24h"],
            "%",
        ),
    )

    st.divider()

    st.header(
        "📈 Sensor History",
        anchor=False,
    )

    history_df = get_sensor_history()

    if history_df is None:

        st.info(
            "N/A — Historical ESP32 sensor data "
            "is not connected yet."
        )

    else:

        fig = px.line(
            history_df,
            x="Time",
            y="Soil Moisture",
            markers=True,
            title="Soil Moisture Over Time",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.header(
        "💦 Water Usage",
        anchor=False,
    )

    water_df = get_water_history()

    if water_df is None:

        st.info(
            "N/A — Historical water usage data "
            "is not connected yet."
        )

    else:

        fig2 = px.bar(
            water_df,
            x="Time",
            y="Water Used (L)",
            title="Water Usage Over Time",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )


# ============================================================
# WEED DETECTION
# ============================================================

elif page == "🌿 Weed Detection":

    st.title(
        "🌿 AI Weed Detection"
    )

    st.write(
        "Upload a crop image or capture a photo "
        "for weed analysis."
    )

    st.divider()

    input_method = st.radio(
        "Image Source",
        [
            "Upload from device",
            "Take a photo",
        ],
        horizontal=True,
    )

    uploaded_file = None

    if input_method == "Upload from device":

        uploaded_file = st.file_uploader(
            "Choose a crop image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )

    else:

        uploaded_file = st.camera_input(
            "Take a crop photo"
        )

    if uploaded_file is not None:

        st.session_state.uploaded_image = (
            uploaded_file
        )

        st.image(
            uploaded_file,
            caption="Crop Image",
            use_container_width=True,
        )

        st.divider()

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True,
        ):

            with st.spinner(
                "Running weed detection..."
            ):

                result = run_weed_detection(
                    uploaded_file
                )

            st.session_state.weed_detected = (
                result["weed_detected"]
            )

            st.session_state.weed_name = (
                result["weed_name"]
            )

            st.session_state.weed_confidence = (
                result["confidence"]
            )

            st.session_state.last_detection_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            st.session_state.recommendation = (
                get_treatment_recommendation(
                    result["weed_name"],
                    crop_name,
                    weather_data,
                )
            )

    st.divider()

    st.header(
        "🤖 Detection Result",
        anchor=False,
    )

    r1, r2 = st.columns(2)

    r1.metric(
        "Weed Status",
        (
            "N/A"
            if st.session_state.weed_detected is None
            else (
                "WEED DETECTED"
                if st.session_state.weed_detected
                else "NO WEED"
            )
        ),
    )

    r2.metric(
        "Confidence",
        display_value(
            st.session_state.weed_confidence,
            " %",
        ),
    )

    st.write(
        "Detected Weed: "
        f"{display_value(st.session_state.weed_name)}"
    )

    if st.session_state.last_detection_time:
        st.caption(
            "Last analysis: "
            f"{st.session_state.last_detection_time}"
        )


# ============================================================
# WEATHER
# ============================================================

elif page == "🌦 Weather":

    title_col, city_col = st.columns(
        [4, 1]
    )

    with title_col:
        st.title(
            "🌦 Weather Monitoring"
        )

    with city_col:
        st.markdown(
            f"### 📍 {WEATHER_CITY}"
        )

    st.write(
        "Current weather and hourly forecast "
        "for agricultural decision making."
    )

    st.caption(
        f"Weather source: {WEATHER_PROVIDER}"
    )

    # --------------------------------------------------------
    # LAST UPDATED
    # --------------------------------------------------------

    if weather_data["fetched_at"] is not None:

        st.caption(
            "Last updated: "
            + weather_data["fetched_at"]
            .astimezone()
            .strftime(
                "%d %b %Y, %I:%M %p"
            )
        )

    else:

        st.caption(
            "Last updated: N/A"
        )

    st.divider()

    # --------------------------------------------------------
    # CURRENT WEATHER
    # --------------------------------------------------------

    st.header(
        "Current Weather",
        anchor=False,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Temperature",
        display_value(
            weather_data["temperature"],
            " °C",
        ),
    )

    c2.metric(
        "Humidity",
        display_value(
            weather_data["humidity"],
            " %",
        ),
    )

    c3.metric(
        "Wind Speed",
        display_value(
            weather_data["wind_speed"],
            " m/s",
        ),
    )

    c4.metric(
        "Condition",
        weather_data["description"],
    )

    st.divider()

    # --------------------------------------------------------
    # RAIN RISK
    # --------------------------------------------------------

    st.header(
        "🌧 Rain Probability",
        anchor=False,
    )

    rs1, rs2, rs3 = st.columns(3)

    rs1.metric(
        "Peak in Next 6 Hours",
        display_value(
            rain_summary["6h"],
            "%",
        ),
    )

    rs2.metric(
        "Peak in Next 12 Hours",
        display_value(
            rain_summary["12h"],
            "%",
        ),
    )

    rs3.metric(
        "Peak in Next 24 Hours",
        display_value(
            rain_summary["24h"],
            "%",
        ),
    )

    st.caption(
        "These values show the highest hourly "
        "precipitation probability within each window."
    )

    st.divider()

    # --------------------------------------------------------
    # HOURLY FORECAST
    # --------------------------------------------------------

    st.header(
        "🌧 Upcoming Forecast",
        anchor=False,
    )

    if not forecast_data:

        st.info(
            "N/A — Forecast data is unavailable."
        )

    else:

        for forecast in forecast_data:

            f1, f2, f3 = st.columns(3)

            formatted_time = forecast[
                "time"
            ].strftime(
                "%d %b, %I:%M %p"
            )

            f1.write(
                f"**Time:** {formatted_time}"
            )

            f2.write(
                f"**Temperature:** "
                f"{forecast['temperature']:.1f} °C"
            )

            f3.write(
                f"**Rain Probability:** "
                f"{forecast['rain_probability']:.0f}%"
            )

            st.write(
                f"**Expected Precipitation:** "
                f"{forecast['rain_amount']:.1f} mm"
            )

            st.write(
                f"**Condition:** "
                f"{forecast['description']}"
            )

            st.divider()


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🧪 Recommendations":

    st.title(
        "🧪 Agricultural Recommendations"
    )

    st.write(
        "This section will combine weed detection, "
        "crop information, AI output, and weather."
    )

    st.divider()

    st.header(
        "🌿 Weed Treatment Recommendation",
        anchor=False,
    )

    r1, r2 = st.columns(2)

    r1.metric(
        "Detected Weed",
        display_value(
            st.session_state.weed_name
        ),
    )

    r2.metric(
        "Crop",
        (
            "N/A"
            if crop_name == "Select crop"
            else crop_name
        ),
    )

    st.write("Treatment: N/A")
    st.write("Dose: N/A")
    st.write("Recommended Spray Window: N/A")
    st.write("Spray Duration: N/A")

    st.info(
        "Recommendation engine: N/A — "
        "AI weed classification and verified "
        "agricultural recommendation data are "
        "not connected yet."
    )


# ============================================================
# AI MODELS
# ============================================================

elif page == "🤖 AI Models":

    st.title(
        "🤖 Irrigation AI Models"
    )

    st.write(
        "Machine-learning and deep-learning models evaluated for irrigation prediction."
    )

    st.divider()

    if get_model_comparison_rows is not None:
        model_comparison = pd.DataFrame(get_model_comparison_rows())
    else:
        model_comparison = pd.DataFrame()

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Model Deep Dive & Live Status"
    )

    model_name = ACTIVE_MODEL_NAME
    st.session_state["selected_model"] = model_name
    trust_info = get_model_trustworthiness(model_name, sensor_data)
    irrigation_prediction = predict_irrigation(
        sensor_data=sensor_data,
        weather_data=weather_data,
        model_name=model_name,
        enforce_complete_readiness=True,
    )

    st.markdown(
        f"<div style='margin-bottom: 16px; padding: 10px 14px; border-radius: 6px; background-color: rgba(245, 127, 23, 0.08); border-left: 4px solid #f57f17; font-size: 0.88rem;'>"
        f"<b>Active Model:</b> {model_name} &nbsp;|&nbsp; <b>Status:</b> <span style='color: #f57f17; font-weight: 600;'>Experimental — Advisory Only</span><br/>"
        f"<span style='font-size: 0.82rem; color: #555;'>Sole active prediction model • 11 environmental features • Tuned Decision Threshold: <code>{trust_info['threshold']:.4f}</code></span><br/>"
        f"<span style='font-size: 0.80rem; color: #b71c1c; font-weight: 500;'>⚠️ Warning: Relative soil moisture from capacitive probes has not been field-validated against dataset scale. Predictions are strictly advisory; automatic pump actuation is disabled.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Detailed inspection breakdown
    c_card1, c_card2 = st.columns(2)
    with c_card1:
        st.markdown("##### 🔬 Verification & Evaluation Profile")
        st.write(f"**Target Task:** {trust_info['target']}")
        st.write(f"**Evaluation Data:** {trust_info['evaluation_type']}")
        st.write(f"**Architecture:** {trust_info['architecture']}")
        st.write(f"**Decision Threshold:** `{trust_info['threshold']:.3f}`")
        if trust_info.get("metrics"):
            st.write("**Evaluation Metrics:**")
            m_items = list(trust_info["metrics"].items())
            m_cols = st.columns(min(len(m_items), 3))
            for i, (k, v) in enumerate(m_items[:6]):
                m_cols[i % min(len(m_items), 3)].metric(k, v)
        st.caption(f"**Evidence Assessment:** {trust_info['summary']}")

    with c_card2:
        st.markdown("##### ⚡ Live Inference Output & Readiness")
        st.write(f"**Readiness:** `{trust_info['input_readiness']}`")
        st.write(f"**Required Inputs:** {trust_info['input_requirements']}")
        if irrigation_prediction["required"] is None:
            st.warning(f"⚠️ **Live Output:** N/A — {irrigation_prediction['message']}")
        elif irrigation_prediction["required"]:
            st.error(f"🚨 **Live Output: Irrigation Needed** (Confidence: {irrigation_prediction['confidence'] * 100:.1f}%) — {irrigation_prediction['message']}")
        else:
            st.success(f"✅ **Live Output: No Irrigation Needed** (Confidence: {(1 - irrigation_prediction['confidence']) * 100:.1f}%) — {irrigation_prediction['message']}")

    # ------------------------------------------------------------
    # 11-FEATURE PROVENANCE & LIVE READINESS AUDIT TABLE
    # ------------------------------------------------------------
    st.divider()

    st.subheader("📋 11-Feature Input Provenance & Live Readiness Audit")
    st.caption(
        "Authoritative 11-feature input schema matching the trained reduced Random Forest model (random_forest_11feat.pkl). "
        "Distinguishes live observed sensors, weather observations, and historically calculated lags. "
        "In accordance with scientific data integrity standards, training medians are never silently substituted as observed live data."
    )

    audit_rf = audit_feature_readiness(sensor_data=sensor_data, weather_data=weather_data, model_name=model_name)
    audit_df = pd.DataFrame(audit_rf["feature_rows"])[
        ["Feature", "Units", "Live Value", "Source", "Provenance", "Ready", "Limitation / Details"]
    ]

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    aud_col1, aud_col2 = st.columns([1, 2])
    with aud_col1:
        ready_pct = (audit_rf["ready_count"] / audit_rf["total_features"]) * 100.0
        st.metric(
            "Features Observed & Ready",
            f"{audit_rf['ready_count']} / {audit_rf['total_features']}",
            f"{ready_pct:.0f}% Complete",
        )
        if audit_rf["is_live_ready"]:
            st.success("✅ Complete 11-Feature Set Ready for Live Prediction (Advisory Only)")
        else:
            st.error(f"🛑 Live Predictions Disabled ({audit_rf['total_features'] - audit_rf['ready_count']} Active Blockers)")

    with aud_col2:
        st.markdown("**Active Readiness Blockers:**")
        if audit_rf["blockers"]:
            for b in audit_rf["blockers"]:
                st.markdown(f"- ⚠️ {b}")
        else:
            st.write("No active blockers detected.")

    # ------------------------------------------------------------
    # EXPERIMENTAL 24H SOIL-MOISTURE FORECASTING (PHASE 13/14/15/22 RESEARCH)
    # ------------------------------------------------------------
    st.divider()

    st.subheader("🧪 Experimental 24-Hour Soil-Moisture Forecasting")
    st.markdown(
        "<div style='margin-bottom: 14px; padding: 10px 14px; border-radius: 6px; "
        "background-color: rgba(211, 47, 47, 0.08); border-left: 4px solid #d32f2f; font-size: 0.90rem;'>"
        "<b>⚠️ EXPERIMENTAL — NOT VALIDATED FOR LIVE FIELD CONTROL</b><br/>"
        "<span style='font-size: 0.84rem; color: #555;'>"
        "Exploratory research track evaluating deep sequence models (LSTM, TCN, Transformer) for 24-hour mean soil-moisture "
        "forecasting (<code>target_mean_24h</code>). These models produce a continuous regression forecast of soil moisture; "
        "they do NOT output an irrigation ON/OFF recommendation. Automatic pump actuation is strictly disabled (<code>PUMP_ALLOWED=False</code>)."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 1. Benchmark Comparison Table
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("##### 📊 Benchmark vs Naive Persistence Baseline")
        st.caption("Evaluated on identical chronological held-out test windows (N = 6,348) across 4 monitored sectors.")
        benchmark_data = {
            "Model": ["Persistence Baseline (Current Moisture)", "Experimental LSTM", "Experimental TCN", "Experimental Transformer"],
            "Parameters": ["0 (Heuristic)", "6,689", "17,921", "18,113"],
            "Test MAE (%)": ["0.412", "3.867", "4.691", "6.124"],
            "Test RMSE (%)": ["0.764", "5.144", "7.698", "8.706"],
            "Pooled R²": ["0.999", "0.938", "0.861", "0.822"],
            "Sector-Level R²": ["Baseline Anchor", "Negative (-7 to -525)", "Negative (-0.2 to -2864)", "Negative (-0.1 to -136)"],
        }
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    with exp_col2:
        st.markdown("##### 🔬 Scientific Audit Findings & Limitations")
        st.markdown(
            "- **Persistence Superiority:** The naive persistence baseline achieves ~10x lower MAE (0.412 vs 3.867/4.691/6.124). "
            "In the post-season test split, irrigation was turned off (0.0% events), leaving soil moisture in static equilibrium.\n"
            "- **Simpson's Paradox in Pooled R²:** Pooled R² appears high (> 0.82) due to large baseline moisture differences "
            "between sectors (2% in Sector 5 to 57% in Sector 4). Within each sector, neural network R² is negative.\n"
            "- **Live Hardware Telemetry Mismatch:** Live ESP32 station lacks EC, pH, solar radiation, and dual-depth soil temperature sensors. "
            "Forecast inference requires 4 hours (24 steps @ 10m) of contiguous 14-channel telemetry. Missing features are never fabricated."
        )

    # 2. Side-by-Side Model Architecture & Status Cards
    st.markdown("##### 🧩 Experimental Forecasting Architectures")
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        st.markdown(
            "<div style='padding: 12px; border-radius: 8px; border: 1px solid #ddd; background-color: #fafafa; min-height: 220px;'>"
            "<b style='color: #2e7d32;'>⚓ Persistence Baseline</b><br/>"
            "<span style='font-size: 0.80rem; color: #666;'>Heuristic Benchmark</span><br/><hr style='margin: 8px 0;'>"
            "<span style='font-size: 0.82rem;'><b>Parameters:</b> 0</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Horizon:</b> 24 Hours</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Input:</b> Current moisture <code>x[t]</code></span><br/>"
            "<span style='font-size: 0.82rem;'><b>Test MAE:</b> 0.412%</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Status:</b> Active Reference Benchmark</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    with card_col2:
        st.markdown(
            "<div style='padding: 12px; border-radius: 8px; border: 1px solid #ddd; background-color: #fafafa; min-height: 220px;'>"
            "<b style='color: #1565c0;'>🧠 Experimental LSTM</b><br/>"
            "<span style='font-size: 0.80rem; color: #666;'>1-Layer LSTM + MLP Head</span><br/><hr style='margin: 8px 0;'>"
            "<span style='font-size: 0.82rem;'><b>Parameters:</b> 6,689</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Architecture:</b> LSTM(14->32), FC(32->16->1)</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Sequence:</b> 24 steps (4h @ 10m)</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Test MAE:</b> 3.867%</span><br/>"
            "<span style='font-size: 0.82rem; color: #2e7d32;'><b>Mode A:</b> Runnable (14-Feat Dataset)</span><br/>"
            "<span style='font-size: 0.82rem; color: #e65100;'><b>Live ESP32:</b> Incompatible (Needs 14-Feat)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    with card_col3:
        st.markdown(
            "<div style='padding: 12px; border-radius: 8px; border: 1px solid #ddd; background-color: #fafafa; min-height: 220px;'>"
            "<b style='color: #6a1b9a;'>⚡ Experimental TCN</b><br/>"
            "<span style='font-size: 0.80rem; color: #666;'>Dilated Causal Conv1D</span><br/><hr style='margin: 8px 0;'>"
            "<span style='font-size: 0.82rem;'><b>Parameters:</b> 17,921</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Architecture:</b> Channels [32,32,32], Dil [1,2,4]</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Sequence:</b> 24 steps (4h @ 10m)</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Test MAE:</b> 4.691%</span><br/>"
            "<span style='font-size: 0.82rem; color: #2e7d32;'><b>Mode A:</b> Runnable (14-Feat Dataset)</span><br/>"
            "<span style='font-size: 0.82rem; color: #e65100;'><b>Live ESP32:</b> Incompatible (Needs 14-Feat)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    with card_col4:
        st.markdown(
            "<div style='padding: 12px; border-radius: 8px; border: 1px solid #ddd; background-color: #fafafa; min-height: 220px;'>"
            "<b style='color: #e65100;'>🔮 Experimental Transformer</b><br/>"
            "<span style='font-size: 0.80rem; color: #666;'>Self-Attention Encoder</span><br/><hr style='margin: 8px 0;'>"
            "<span style='font-size: 0.82rem;'><b>Parameters:</b> 18,113</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Architecture:</b> 2 Layers, 4 Heads, d_model=32</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Sequence:</b> 24 steps (4h @ 10m)</span><br/>"
            "<span style='font-size: 0.82rem;'><b>Test MAE:</b> 6.124%</span><br/>"
            "<span style='font-size: 0.82rem; color: #2e7d32;'><b>Mode A:</b> Runnable (14-Feat Dataset)</span><br/>"
            "<span style='font-size: 0.82rem; color: #e65100;'><b>Live ESP32:</b> Incompatible (Needs 14-Feat)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # 3. Two Accessible Forecasting Modes: Offline Research Mode & Live Hardware Mode
    st.markdown("##### 🚀 Model Forecasting Execution Controls")
    
    forecasting_mode = st.radio(
        "Select Forecasting Mode:",
        ["🔬 Mode A: Offline Research Mode (Compatible Stored Dataset)", "📡 Mode B: Live Hardware Mode (Real ESP32 Telemetry)"],
        horizontal=True,
        key="forecasting_mode_radio",
    )

    if "Mode A" in forecasting_mode:
        st.info(
            "🔬 **Mode A — Offline Research Mode:** Runs real inference using genuine 24-step (4-hour) historical sequences "
            "from the multi-sector precision irrigation research dataset. All 14 physical channels are genuinely present. "
            "Results are continuous 24-hour mean soil moisture regression forecasts (`target_mean_24h`), not live hardware predictions.",
            icon="ℹ️",
        )
        mode_a_col1, mode_a_col2 = st.columns([2, 1])
        with mode_a_col1:
            sample_sector = st.selectbox(
                "Select Stored Evaluation Sector Sequence:",
                options=[1, 2, 4, 5],
                format_func=lambda z: {
                    1: "Sector 1 — Kiwi Orchard (Target mean: ~19%)",
                    2: "Sector 2 — Vineyard (Target mean: ~15%)",
                    4: "Sector 4 — Zucchini (Target mean: ~55%)",
                    5: "Sector 5 — Blueberry (Target mean: ~0%)",
                }.get(z, f"Sector {z}"),
                key="sample_forecasting_sector",
            )
        with mode_a_col2:
            model_to_run = st.selectbox(
                "Select Model to Inspect:",
                ["All Models (Comparative)", "LSTM Only", "TCN Only", "Transformer Only"],
                key="mode_a_model_select",
            )

        try:
            sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase13"))
            sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"))
            from forecasting_models import (
                load_forecasting_model,
                predict_future_moisture,
                EXPERIMENTAL_FEATURES,
            )
            from transformer_inference import (
                load_transformer_model,
                predict_transformer_future_moisture,
            )

            csv_sample_path = (
                PROJECT_ROOT
                / "Soil Moisture, Irrigation Actuator and Weather Dat"
                / "02_processed_data"
                / "preprocessed"
                / f"dataset_zone_{sample_sector}_preprocessed.csv"
            )

            if csv_sample_path.exists():
                sample_df = pd.read_csv(csv_sample_path).iloc[-24:].reset_index(drop=True)
                lstm_model, lstm_meta = load_forecasting_model("LSTM")
                tcn_model, tcn_meta = load_forecasting_model("TCN")
                trans_model, trans_meta = load_transformer_model()

                lstm_fc = predict_future_moisture(lstm_model, sample_df, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"])
                tcn_fc = predict_future_moisture(tcn_model, sample_df, tcn_meta["scaler_mean"], tcn_meta["scaler_scale"])
                trans_fc = predict_transformer_future_moisture(trans_model, sample_df, trans_meta["scaler_mean"], trans_meta["scaler_scale"])
                persist_val = float(sample_df["soil_moisture"].iloc[-1])
                actual_target = float(sample_df["target_mean_24h"].iloc[-1]) if "target_mean_24h" in sample_df.columns else None

                if model_to_run == "All Models (Comparative)":
                    inf_col1, inf_col2, inf_col3, inf_col4 = st.columns(4)
                    inf_col1.metric("⚓ Persistence Baseline", f"{persist_val:.2f}%", help="Last observed moisture in 24-step sequence")
                    inf_col2.metric("🧠 LSTM Forecast", f"{lstm_fc:.2f}%", delta=f"{lstm_fc - persist_val:+.2f}% vs persist")
                    inf_col3.metric("⚡ TCN Forecast", f"{tcn_fc:.2f}%", delta=f"{tcn_fc - persist_val:+.2f}% vs persist")
                    inf_col4.metric("🔮 Transformer Forecast", f"{trans_fc:.2f}%", delta=f"{trans_fc - persist_val:+.2f}% vs persist")
                elif model_to_run == "LSTM Only":
                    st.metric("🧠 Experimental LSTM 24h Forecast", f"{lstm_fc:.2f}%", delta=f"{lstm_fc - persist_val:+.2f}% vs persist ({persist_val:.2f}%)")
                    st.caption("Architecture: LSTM(14->32), FC(32->16->1) • 6,689 Parameters • Target: target_mean_24h")
                elif model_to_run == "TCN Only":
                    st.metric("⚡ Experimental TCN 24h Forecast", f"{tcn_fc:.2f}%", delta=f"{tcn_fc - persist_val:+.2f}% vs persist ({persist_val:.2f}%)")
                    st.caption("Architecture: Dilated Causal Conv1D (channels=[32,32,32]) • 17,921 Parameters • Target: target_mean_24h")
                elif model_to_run == "Transformer Only":
                    st.metric("🔮 Experimental Transformer 24h Forecast", f"{trans_fc:.2f}%", delta=f"{trans_fc - persist_val:+.2f}% vs persist ({persist_val:.2f}%)")
                    st.caption("Architecture: 2-Layer TransformerEncoder (4 heads, d_model=32) • 18,113 Parameters • Target: target_mean_24h")

                if actual_target is not None:
                    st.markdown(f"🎯 **Ground Truth Target (`target_mean_24h`):** `{actual_target:.2f}%` &nbsp;|&nbsp; Sequence Horizon: 24 steps @ 10-minute cadence (4 hours history)")
                
                with st.expander("🔍 View 24-Step Stored Sequence Input Data (Last 5 timesteps)"):
                    st.dataframe(sample_df[EXPERIMENTAL_FEATURES].tail(5), use_container_width=True)
            else:
                st.warning(f"Stored evaluation dataset file not found at: {csv_sample_path.name}")
        except Exception as e:
            st.error(f"Error loading experimental forecasting models: {e}")

    else:
        # Mode B: Live Hardware Mode & Live Trend Forecast
        st.markdown("###### 📡 Live ESP32 Soil Moisture Forecast (Experimental Trend-Based)")
        st.caption(
            "Produces a live short-term soil moisture trend projection exclusively from genuine ESP32 telemetry "
            "and SQLite observation history (`ten_minute_observations`). Expressed as relative moisture index (0-100%), "
            "explicitly NOT volumetric water content (VWC). Not a trained neural network or machine learning model."
        )

        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental"))
        from live_forecast import (
            generate_live_esp32_forecast,
            fetch_optional_weather_forecast,
        )

        # Select Live Forecasting Strategy
        st.markdown("###### 🎛️ Live Soil-Moisture Forecast Strategy")
        forecast_strategy = st.radio(
            "Select Live Forecasting Strategy:",
            ["📈 Trend Forecast", "🧠 Experimental LSTM", "⚖️ Model Comparison"],
            horizontal=True,
            key="live_forecast_submode_radio",
        )

        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "live_soil_forecaster"))
        from forecasting_pipeline import get_forecast_by_mode

        if forecast_strategy == "📈 Trend Forecast":
            # Controls: Forecast Horizon & History Window
            fc_ctrl_col1, fc_ctrl_col2 = st.columns([1, 1])
            with fc_ctrl_col1:
                horizon_sel = st.selectbox(
                    "Forecast Horizon:",
                    options=[15, 30, 45, 60],
                    index=1,  # Default 30 min
                    format_func=lambda h: f"{h} Minutes Ahead",
                    key="live_fc_horizon_select",
                )
            with fc_ctrl_col2:
                window_sel = st.selectbox(
                    "History Observation Window:",
                    options=[60, 120, 240],
                    index=1,  # Default 120 min (2h)
                    format_func=lambda w: f"Last {w} Minutes ({w//60} Hours)",
                    key="live_fc_window_select",
                )

            # Run Live Forecast Pipeline
            live_fc = generate_live_esp32_forecast(
                horizon_minutes=horizon_sel,
                window_minutes=window_sel,
            )

            if live_fc.get("can_forecast"):
                p1 = live_fc["probe1"]
                p2 = live_fc["probe2"]
                avg = live_fc["average"]
                conf = live_fc["confidence"]
                conf_badge = {"HIGH": "🟢 High Confidence", "MEDIUM": "🟡 Medium Confidence", "LOW": "🟠 Low Confidence"}.get(conf, "⚪ Unknown")

                # Metrics row
                fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
                p1_curr_str = f"{p1['current_index']:.1f}%" if p1.get('calibrated') else "Uncalibrated"
                p1_proj_str = f"{p1['projected_index']:.1f}%" if p1.get('calibrated') else "N/A"
                p1_delta = f"{p1['rate_per_hour']:+.2f}%/hr ({p1['direction']})" if p1.get('calibrated') else None

                p2_curr_str = f"{p2['current_index']:.1f}%" if p2.get('calibrated') else "Uncalibrated"
                p2_proj_str = f"{p2['projected_index']:.1f}%" if p2.get('calibrated') else "N/A"
                p2_delta = f"{p2['rate_per_hour']:+.2f}%/hr ({p2['direction']})" if p2.get('calibrated') else None

                avg_curr_str = f"{avg['current_index']:.1f}%" if avg.get('current_index') is not None else "N/A"
                avg_proj_str = f"{avg['projected_index']:.1f}%" if avg.get('projected_index') is not None else "N/A"
                avg_delta = f"{avg['rate_per_hour']:+.2f}%/hr ({avg['direction']})" if avg.get('rate_per_hour') is not None else None

                # Live Station Readings & Freshness Context
                freshness_str = f"{live_fc['time_since_latest_minutes']:.1f}m ago" if live_fc.get("time_since_latest_minutes") is not None else "Unknown"
                freshness_badge = "🟢 Fresh (<30m)" if not live_fc.get("is_stale") else f"🟠 Stale ({freshness_str})"

                st.markdown(
                    f"<div style='margin-bottom: 12px; padding: 10px 14px; border-radius: 6px; background-color: #f8f9fa; border: 1px solid #e0e0e0; font-size: 0.85rem;'>"
                    f"<b>📡 Live Edge Telemetry Status:</b> &nbsp;|&nbsp; <b>Data Freshness:</b> {freshness_badge} &nbsp;|&nbsp; <b>Last Timestamp:</b> <code>{live_fc['latest_timestamp_utc']}</code><br/>"
                    f"<span style='color: #444; font-size: 0.82rem;'><b>Latest Air Temp:</b> {live_fc.get('latest_temperature_c', 'N/A')}°C &nbsp;|&nbsp; <b>Latest Humidity:</b> {live_fc.get('latest_humidity_pct', 'N/A')}% &nbsp;|&nbsp; "
                    f"<b>Calibration:</b> Probe 1 ({'🟢 Calibrated' if p1.get('calibrated') else '🔴 Raw ADC'}), Probe 2 ({'🟢 Calibrated' if p2.get('calibrated') else '🔴 Raw ADC'})</span><br/>"
                    f"<span style='color: #b71c1c; font-weight: 500; font-size: 0.80rem;'>⚠️ Note: Values represent relative air/water capacitive index (0.0% - 100.0%), strictly NOT volumetric water content (VWC).</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                fc_col1.metric("📍 Probe 1 (Current / Trend)", p1_curr_str, delta=p1_delta)
                fc_col2.metric("📍 Probe 2 (Current / Trend)", p2_curr_str, delta=p2_delta)
                fc_col3.metric(f"🔮 Average {horizon_sel}m Projected Index", avg_proj_str, delta=avg_delta)
                fc_col4.metric("🛡️ Data Quality", conf_badge, help=live_fc.get("confidence_reason"))

                st.caption(
                    f"ℹ️ **Observation Window:** Based on `{live_fc['observations_count']}` continuous 10-minute observations "
                    f"spanning `{live_fc['streak_duration_minutes']:.0f}` minutes up to `{live_fc['latest_timestamp_utc']}`. "
                    f"Forecast Horizon: `{horizon_sel} minutes` ahead. Uncertainty: `±{avg['uncertainty_pm']:.1f}%` relative index. {live_fc.get('confidence_reason')}"
                )

                # Historical trend chart with projection
                if "history_df" in live_fc and not live_fc["history_df"].empty:
                    import plotly.graph_objects as go
                    hdf = live_fc["history_df"]
                    fig_fc = go.Figure()

                    if p1.get("calibrated"):
                        fig_fc.add_trace(go.Scatter(
                            x=hdf["interval_start_utc"],
                            y=hdf["probe1_index"],
                            mode="lines+markers",
                            name="Probe 1 Relative Index (%)",
                            line=dict(color="#1f77b4", width=2),
                        ))
                    if p2.get("calibrated"):
                        fig_fc.add_trace(go.Scatter(
                            x=hdf["interval_start_utc"],
                            y=hdf["probe2_index"],
                            mode="lines+markers",
                            name="Probe 2 Relative Index (%)",
                            line=dict(color="#2ca02c", width=2),
                        ))

                    # Add projected forecast point
                    last_utc_dt = pd.to_datetime(live_fc["latest_timestamp_utc"])
                    future_utc_dt = last_utc_dt + pd.Timedelta(minutes=horizon_sel)
                    future_utc_str = future_utc_dt.isoformat()

                    if avg.get("projected_index") is not None:
                        fig_fc.add_trace(go.Scatter(
                            x=[hdf["interval_start_utc"].iloc[-1], future_utc_str],
                            y=[avg["current_index"], avg["projected_index"]],
                            mode="lines+markers",
                            name=f"Projected Trend ({horizon_sel}m ahead)",
                            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                            marker=dict(size=[6, 10], symbol=["circle", "star"]),
                        ))

                    fig_fc.update_layout(
                        title="Live ESP32 Soil Moisture History & Short-Term Trend Projection",
                        xaxis_title="UTC Timestamp",
                        yaxis_title="Relative Moisture Index (%)",
                        yaxis=dict(range=[0, 105]),
                        margin=dict(l=30, r=30, t=40, b=30),
                        height=320,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_fc, width="stretch")

            else:
                # Cannot forecast
                st.warning(
                    f"⚠️ **Live Forecast Unavailable:** {live_fc['message']}\n\n"
                    f"**Reason:** `{live_fc.get('status')}`. {live_fc.get('disclaimer')}",
                    icon="⚠️",
                )

        elif forecast_strategy == "🧠 Experimental LSTM":
            lstm_res = get_forecast_by_mode("Experimental LSTM")
            if lstm_res.get("available"):
                st.metric("🧠 Experimental LSTM 60m Forecast", f"{lstm_res['projected_index']:.1f}%")
                st.caption(f"Forecast Horizon: 60 Minutes Ahead • Target: {lstm_res.get('target_definition')}")
            else:
                st.warning(
                    f"🔒 **{lstm_res['status']}**\n\n"
                    f"{lstm_res.get('reason')}\n\n"
                    f"- **Forecast Horizon:** `60 minutes ahead (6 continuous 10m intervals)`\n"
                    f"- **Target Definition:** `{lstm_res.get('target_definition')}`\n"
                    f"- **Sample Status:** `{lstm_res.get('samples_status')}`\n"
                    f"- **Active Operational Baseline:** `{lstm_res.get('operational_baseline')}`\n\n"
                    f"💡 *In accordance with zero-fabrication rules, model training is withheld until multi-day telemetry accumulates.*",
                    icon="🔒",
                )

        elif forecast_strategy == "⚖️ Model Comparison":
            comp_res = get_forecast_by_mode("Model Comparison", horizon_minutes=60, window_minutes=120)
            c1, c2, c3, c4 = st.columns(4)

            p_val = f"{comp_res['persistence_baseline']['projected_index']:.1f}%" if comp_res['persistence_baseline']['available'] else "N/A"
            c1.metric("⚓ Persistence Baseline", p_val, help="Last observed average moisture index")

            t_val = f"{comp_res['trend_baseline']['projected_index']:.1f}%" if comp_res['trend_baseline']['available'] else "N/A"
            t_rate = f"{comp_res['trend_baseline']['rate_per_hour']:+.2f}%/hr" if comp_res['trend_baseline'].get('rate_per_hour') is not None else None
            c2.metric("📈 Trend Baseline (60m)", t_val, delta=t_rate, help="Linear trend extrapolation")

            lstm_val = "🔒 Withheld" if not comp_res['experimental_lstm']['available'] else f"{comp_res['experimental_lstm']['projected_index']:.1f}%"
            c3.metric("🧠 Experimental LSTM", lstm_val, help=comp_res['experimental_lstm']['status'])

            c4.metric("🎯 Recommended Forecaster", comp_res['operational_recommendation'], help="Model acceptance gate recommendation")

            st.caption(
                f"ℹ️ **Model Acceptance Gate Policy:** {comp_res['acceptance']['reason']}"
            )

        # Optional Weather Context from Open-Meteo
        wx = fetch_optional_weather_forecast(latitude=WEATHER_LATITUDE, longitude=WEATHER_LONGITUDE)
        with st.expander("🌤️ Optional Real-Time Atmospheric Weather Context (Open-Meteo)", expanded=False):
            if WEATHER_LATITUDE is None or WEATHER_LONGITUDE is None:
                st.warning("⚠️ Location coordinates unconfigured. Please configure `WEATHER_LATITUDE` and `WEATHER_LONGITUDE`.")
            elif wx.get("available"):
                w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                w_col1.metric("Air Temp (2m)", f"{wx['temperature_2m_c']:.1f}°C")
                w_col2.metric("Relative Humidity", f"{wx['relative_humidity_2m_pct']:.0f}%")
                w_col3.metric("Rain Prob (Next Hr)", f"{wx['precipitation_probability_pct'] or 0:.0f}%")
                w_col4.metric("Conditions", wx['weather_description'])
                st.caption(f"📌 Location: `{WEATHER_CITY}` ({WEATHER_LATITUDE}, {WEATHER_LONGITUDE}). {wx['disclaimer']}")
            else:
                st.info(f"Atmospheric forecast unavailable: {wx.get('error')}")

        st.markdown("---")
        # 14-Feature Input Verification Test for LSTM/TCN/Transformer
        st.markdown("###### 🔍 14-Feature Model Schema Compatibility Diagnostic")
        st.caption(
            "Rigorous audit of input feasibility against the 14-feature sequence model schema. "
            "Distinguishes direct ESP32 measurements, external weather, model-derived estimates, and missing physical channels."
        )

        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase25"))
        from realtime_history import extract_realtime_24step_hardware_sequence
        from live_forecast import fetch_hourly_weather_observations, align_weather_to_observation_timestamps

        hw_seq = extract_realtime_24step_hardware_sequence()
        hourly_wx = fetch_hourly_weather_observations(latitude=WEATHER_LATITUDE, longitude=WEATHER_LONGITUDE)

        hw_col1, hw_col2 = st.columns([1, 1])
        with hw_col1:
            st.write(f"**Live Observation Buffer:** {'🟢 Available (24 continuous steps)' if hw_seq['available'] else '🟡 Accumulating'}")
            st.write(f"**Longest Continuous Streak:** `{hw_seq['streak_length']}` observations ({hw_seq['streak_length']*10/60:.1f} hours)")
            if sensor_data:
                st.write(f"**Latest Live Raw Probe 1:** `{sensor_data.get('soil1_raw', 'N/A')}` ADC")
                st.write(f"**Latest Live Raw Probe 2:** `{sensor_data.get('soil2_raw', 'N/A')}` ADC")
                st.write(f"**Latest Live Air Temp / Humidity:** `{sensor_data.get('temperature', 'N/A')}°C` / `{sensor_data.get('humidity', 'N/A')}%`")
        with hw_col2:
            st.write("**Required Model Schema:** 14 physical features @ 24 timesteps (4h window)")
            st.write(f"**External Weather Integration:** {'🟢 Connected (Open-Meteo)' if hourly_wx.get('available') else '🔴 Offline / Unavailable'}")
            if hourly_wx.get("available") and hourly_wx.get("weather_df") is not None:
                latest_w = hourly_wx["weather_df"].iloc[-1]
                st.write(f"**Latest Pressure / Wind / Rad:** `{latest_w['weather_pressure']:.1f} hPa` / `{latest_w['weather_wind_speed']:.1f} km/h` / `{latest_w['weather_radiation']:.1f} W/m²`")
            st.write("**Missing In-Situ Physical Channels:** 4 channels (`ec`, `ph`, `soil_temp_7-18cm`, `irrigation_duration`)")

        # Detailed 14-Feature Categorization Diagnostic Table
        with st.expander("📊 14-Feature Categorization & Feasibility Diagnostic Table", expanded=True):
            feasibility_audit = [
                {
                    "Feature": "weather_temp",
                    "Required Units": "°C",
                    "Category": "Direct ESP32 Measurement",
                    "Source": "ESP32 DHT22 (GPIO 4)",
                    "Status": "✅ Available (In-Situ)",
                    "Scientific Assessment": "Direct on-site ambient air temperature measurement.",
                },
                {
                    "Feature": "weather_humidity",
                    "Required Units": "%",
                    "Category": "Direct ESP32 Measurement",
                    "Source": "ESP32 DHT22 (GPIO 4)",
                    "Status": "✅ Available (In-Situ)",
                    "Scientific Assessment": "Direct on-site relative humidity measurement.",
                },
                {
                    "Feature": "weather_rain",
                    "Required Units": "mm / 10min",
                    "Category": "Unit / Calibration Mismatch",
                    "Source": "ESP32 Rain Board (GPIO 27)",
                    "Status": "⚠️ Incompatible Scale",
                    "Scientific Assessment": "Board senses qualitative surface moisture (0/1), not quantitative rainfall volume (mm).",
                },
                {
                    "Feature": "weather_pressure",
                    "Required Units": "hPa",
                    "Category": "External Weather Observation",
                    "Source": f"Open-Meteo ({WEATHER_CITY})",
                    "Status": "🌐 Sourced Externally",
                    "Scientific Assessment": "Hourly surface pressure (hPa) aligned via UTC linear interpolation without extrapolation.",
                },
                {
                    "Feature": "weather_wind_speed",
                    "Required Units": "km/h",
                    "Category": "External Weather Observation",
                    "Source": f"Open-Meteo ({WEATHER_CITY})",
                    "Status": "🌐 Sourced Externally",
                    "Scientific Assessment": "Hourly 10m wind speed (km/h) aligned via UTC linear interpolation without extrapolation.",
                },
                {
                    "Feature": "weather_radiation",
                    "Required Units": "W/m²",
                    "Category": "External Weather Observation",
                    "Source": f"Open-Meteo ({WEATHER_CITY})",
                    "Status": "🌐 Sourced Externally",
                    "Scientific Assessment": "Direct solar radiation (W/m²) matching training definition.",
                },
                {
                    "Feature": "soil_temperature_0-7cm",
                    "Required Units": "°C",
                    "Category": "Model-Derived Soil Estimate",
                    "Source": f"Open-Meteo ERA5 Layer ({WEATHER_CITY})",
                    "Status": "🌐 Model-Derived Estimate",
                    "Scientific Assessment": "Simulated shallow land-surface layer (0-7cm) from NWP/ERA5 reanalysis, not in-situ soil probe.",
                },
                {
                    "Feature": "soil_temperature_7-18cm",
                    "Required Units": "°C",
                    "Category": "Missing Physical Measurement",
                    "Source": "None on ESP32",
                    "Status": "❌ Missing Channel",
                    "Scientific Assessment": "Subsoil probe at 7-18cm absent. Cannot substitute air DHT22 or arbitrary constant.",
                },
                {
                    "Feature": "ec",
                    "Required Units": "µS/cm",
                    "Category": "Missing Physical Measurement",
                    "Source": "None on ESP32",
                    "Status": "❌ Missing Channel",
                    "Scientific Assessment": "Soil electrical conductivity probe absent from hardware rig. Zero fabrication policy.",
                },
                {
                    "Feature": "ph",
                    "Required Units": "pH (0-14)",
                    "Category": "Missing Physical Measurement",
                    "Source": "None on ESP32",
                    "Status": "❌ Missing Channel",
                    "Scientific Assessment": "Soil pH electrode absent from hardware rig. Zero fabrication policy.",
                },
                {
                    "Feature": "soil_moisture",
                    "Required Units": "% (VWC)",
                    "Category": "Unit / Calibration Mismatch",
                    "Source": "Capacitive Probes (GPIO 32/33)",
                    "Status": "⚠️ Uncalibrated Index",
                    "Scientific Assessment": "Outputs relative 0-100% air/water index, not volumetric water content (VWC).",
                },
                {
                    "Feature": "irrigation_duration_minutes",
                    "Required Units": "min (0-10)",
                    "Category": "Missing Physical Measurement",
                    "Source": "None (PUMP_ALLOWED=False)",
                    "Status": "❌ Missing Actuator Telemetry",
                    "Scientific Assessment": "Pump is hard-locked. No valve telemetry or flow meter. Reservoir depth != irrigation duration.",
                },
                {
                    "Feature": "hour",
                    "Required Units": "0-23",
                    "Category": "Direct ESP32 Measurement",
                    "Source": "Observation UTC Timestamp",
                    "Status": "✅ Deterministic",
                    "Scientific Assessment": "Deterministic temporal coordinate derived from host observation clock.",
                },
                {
                    "Feature": "month",
                    "Required Units": "1-12",
                    "Category": "Direct ESP32 Measurement",
                    "Source": "Observation UTC Timestamp",
                    "Status": "✅ Deterministic",
                    "Scientific Assessment": "Deterministic temporal coordinate derived from host observation clock.",
                },
            ]
            st.dataframe(pd.DataFrame(feasibility_audit), use_container_width=True, hide_index=True)

        # Honest Neural Network Inference Guardrail
        st.markdown("**14-Feature Live Sequence Model Execution Attempt:**")
        if hw_seq["available"] and hw_seq["sequence_df"] is not None:
            df_hw_candidate = hw_seq["sequence_df"]
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase13"))
                from forecasting_models import load_forecasting_model, predict_future_moisture
                lm, lmeta = load_forecasting_model("LSTM")
                # Strictly attempt inference with genuine input validation
                predict_future_moisture(lm, df_hw_candidate, lmeta["scaler_mean"], lmeta["scaler_scale"])
                st.success("Live inference executed successfully.")
            except ValueError as ve:
                st.warning(
                    f"⚠️ **Live Neural Inference Withheld:** Input validation safely rejected live hardware sequence: `{ve}`.\n\n"
                    "**Why execution remains blocked:** Even with external weather variables (pressure, wind, solar radiation) integrated, "
                    "in-situ physical channels (`soil_temperature_7-18cm`, `ec`, `ph`, and `irrigation_duration_minutes`) cannot be measured "
                    "by the ESP32 station. In accordance with zero-fabrication constraints, missing channels are NOT filled with dummy numbers. "
                    "Use **Mode A** above to run the neural checkpoints on fully compatible research sequences, or use the **Live Trend Forecast** above.",
                    icon="⚠️",
                )
        else:
            st.warning(
                f"⚠️ **Live Hardware Buffer Accumulating:** {hw_seq['blocker']}\n\n"
                "Once 24 continuous 10-minute intervals are reached, the hardware sequence becomes available. "
                "Meanwhile, all three models remain 100% accessible and executable in **Mode A (Offline Research Mode)** above.",
                icon="⚠️",
            )

        st.markdown("---")
        # Separate Live Neural Forecaster (Experimental 6-Feature Reduced-Input)
        st.markdown("###### 🧠 Live Neural Forecasting Pipeline (Experimental 6-Feature Reduced-Input)")
        st.caption(
            "Separate live neural forecasting architecture designed exclusively for genuine ESP32 telemetry. "
            "Uses a reduced 6-feature schema (calibrated probes 1 & 2, DHT22 temp & humidity, UTC hour & month) "
            "and targets 60-minute future relative moisture index, eliminating dependencies on missing subsoil/EC/pH channels."
        )

        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "live_soil_forecaster"))
        from forecasting_pipeline import (
            get_live_neural_forecast_status,
            assess_dataset_training_sufficiency,
            LIVE_FEATURE_SCHEMA,
            TARGET_DEFINITION,
            TARGET_HORIZON_MINUTES,
            TARGET_UNITS,
        )

        nf_status = get_live_neural_forecast_status()
        audit_res = assess_dataset_training_sufficiency()

        nf_col1, nf_col2, nf_col3, nf_col4 = st.columns(4)
        nf_col1.metric("Architecture", "6-Feature Reduced", help=", ".join(LIVE_FEATURE_SCHEMA))
        nf_col2.metric("Forecast Horizon", f"{TARGET_HORIZON_MINUTES} min (6 steps)", help="10-minute aggregation cadence")
        nf_col3.metric("Valid Pairs in DB", f"{audit_res['valid_sequences_count']} / 200 min", help="Continuous (6-step input, 6-step future target) pairs")
        nf_col4.metric("Model Status", "🔒 Training Withheld", help="Withheld due to insufficient sample count")

        st.warning(
            f"🔒 **Live Neural Forecaster Disabled (Scientific Integrity Guardrail):**\n\n"
            f"{nf_status['blocker']}\n\n"
            f"- **Target Definition:** `{TARGET_DEFINITION}` ({TARGET_UNITS})\n"
            f"- **Target Horizon:** `{TARGET_HORIZON_MINUTES} minutes ahead` (6 steps @ 10-minute cadence)\n"
            f"- **Artifact Path:** `{nf_status['artifact_path']}` (uninstantiated until dataset is sufficient)\n"
            f"- **Active Operational Baseline:** `{nf_status['operational_baseline']}`\n\n"
            f"**Zero-Fabrication Guarantee:** Deep neural sequence models require multi-day dry-down cycles to avoid severe memorization. "
            f"The empirical trend extrapolation remains active above and serves as the operational baseline.",
            icon="🔒",
        )

        with st.expander("🔬 6-Feature Live Telemetry Schema & Target Specifications", expanded=False):
            schema_details = [
                {"Index": 1, "Feature": "soil1_relative_pct", "Units": "% (0-100)", "Physical Source": "Probe 1 (GPIO 32, dry 2054, wet 1073 ADC)", "Type": "Calibrated Relative Index"},
                {"Index": 2, "Feature": "soil2_relative_pct", "Units": "% (0-100)", "Physical Source": "Probe 2 (GPIO 33, dry 1906, wet 1023 ADC)", "Type": "Calibrated Relative Index"},
                {"Index": 3, "Feature": "temperature_c", "Units": "°C", "Physical Source": "DHT22 (GPIO 4 ambient air temperature)", "Type": "Direct Measurement"},
                {"Index": 4, "Feature": "humidity_pct", "Units": "%", "Physical Source": "DHT22 (GPIO 4 relative humidity)", "Type": "Direct Measurement"},
                {"Index": 5, "Feature": "hour", "Units": "0-23", "Physical Source": "Observation Host Clock (UTC)", "Type": "Deterministic Temporal Coordinate"},
                {"Index": 6, "Feature": "month", "Units": "1-12", "Physical Source": "Observation Host Clock (UTC)", "Type": "Deterministic Temporal Coordinate"},
            ]
            st.dataframe(pd.DataFrame(schema_details), width="stretch", hide_index=True)
            st.info(
                "💡 **Relative Index vs. Volumetric Water Content (VWC):**\n"
                "The calibrated probe values represent capacitive impedance relative to dry air and distilled water reference points. "
                "They do NOT measure volumetric soil water fraction (m³/m³ or % VWC). "
                "The 14-feature models require gravimetrically calibrated VWC and cannot accept relative capacitive index.",
                icon="ℹ️",
            )

    # 4. Phase 24 Live Sensor Data Readiness & Calibration Pipeline
    st.markdown("##### 📡 Live Telemetry & 14-Feature Input Readiness Audit (Phase 24)")
    st.caption(
        "Evaluates real ESP32 edge telemetry, SQLite database coverage, probe calibration status, "
        "and physical compatibility against the 14-feature sequence forecasting schema."
    )

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase24"))
        from data_collection import audit_sqlite_data_collection
        coll_audit = audit_sqlite_data_collection()
        feat_audit = coll_audit["feature_readiness"]

        # 4a. Telemetry Coverage Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Total Raw Samples", f"{coll_audit['total_raw_samples']:,}", help="Total raw serial packets in sensor_readings table")
        m2.metric("⏱️ 10-Min Intervals", f"{coll_audit['aggregated_intervals_count']}", help="Aggregated 10-minute intervals")
        m3.metric("⚠️ Disconnect Gaps", f"{len(coll_audit['gaps'])}", help=f"Total missing intervals: {coll_audit['total_missing_intervals']}")
        m4.metric("🕒 Telemetry Span", f"{coll_audit['total_span_hours']:.1f} hrs", help="Total hours between earliest and latest recorded packets")

        # 4b. Probe Calibration Status
        p1_cal = coll_audit["probes_calibration"].get("probe1", {})
        p2_cal = coll_audit["probes_calibration"].get("probe2", {})
        cal_col1, cal_col2 = st.columns(2)
        with cal_col1:
            st.markdown(
                f"<div style='padding: 8px 12px; border-radius: 6px; border: 1px solid #ddd; background-color: #f9f9f9; font-size: 0.85rem;'>"
                f"<b>Probe 1 (GPIO 32):</b> "
                f"{'🟢 Calibrated (Provisional Relative %)' if p1_cal.get('calibrated') else '🔴 Uncalibrated (Raw ADC only)'}<br/>"
                f"<span style='color: #666; font-size: 0.80rem;'>Dry: <code>{p1_cal.get('dry_adc', 'N/A')}</code> | Wet: <code>{p1_cal.get('wet_adc', 'N/A')}</code> | "
                f"Status: Air/water index, not volumetric water content (VWC).</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with cal_col2:
            st.markdown(
                f"<div style='padding: 8px 12px; border-radius: 6px; border: 1px solid #ddd; background-color: #f9f9f9; font-size: 0.85rem;'>"
                f"<b>Probe 2 (GPIO 33):</b> "
                f"{'🟢 Calibrated (Provisional Relative %)' if p2_cal.get('calibrated') else '🔴 Uncalibrated (Raw ADC only)'}<br/>"
                f"<span style='color: #666; font-size: 0.80rem;'>Dry: <code>{p2_cal.get('dry_adc', 'N/A')}</code> | Wet: <code>{p2_cal.get('wet_adc', 'N/A')}</code> | "
                f"Status: Air/water index, not volumetric water content (VWC).</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 4c. Feature Readiness Breakdown Table
        feat_df = pd.DataFrame(feat_audit["rows"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # 4d. Mandatory Live Forecast Status Banner
        st.error(
            f"🛑 **{feat_audit['mandatory_status_label']}**\n\n"
            f"**Readiness Summary:** {feat_audit['category_counts']['Available from genuine live data']} / 14 features available • "
            f"{feat_audit['category_counts']['Requires calibration']} requiring soil core calibration • "
            f"{feat_audit['category_counts']['Available but incompatible in scale or meaning']} incompatible scale/meaning • "
            f"{feat_audit['category_counts']['Missing']} completely missing physical channels.\n\n"
            f"Experimental live forecasting remains strictly disabled until physical weather/soil telemetry, gravimetric soil calibration, "
            f"and contiguous 4-hour historical buffers are implemented. Missing sensor measurements are strictly never fabricated.",
            icon="🛑",
        )

        # 4e. Phase 25 Real-Time Hardware History Infrastructure Status
        sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase25"))
        from realtime_history import (
            sync_raw_readings_to_ten_minute_observations,
            extract_realtime_24step_hardware_sequence,
        )
        sync_res = sync_raw_readings_to_ten_minute_observations()
        hw_seq_res = extract_realtime_24step_hardware_sequence()

        st.markdown("##### ⚡ Real-Time Hardware History Infrastructure (Phase 25)")
        st.caption(
            "Maintains synchronized ten-minute observations in SQLite (`ten_minute_observations` table). "
            "Evaluates whether a continuous 24-step (4.0-hour) sequence of genuine ESP32 telemetry is available."
        )

        h1, h2, h3 = st.columns(3)
        h1.metric("📦 Synchronized 10m Buckets", f"{sync_res['buckets_written']}")
        h2.metric("🔗 Longest Continuous Streak", f"{hw_seq_res['streak_length']} steps", f"{hw_seq_res['streak_length']*10/60:.1f} hrs continuous")
        if hw_seq_res["available"]:
            h3.success(f"🟢 4-Hour Hardware Buffer Ready ({hw_seq_res['streak_length']} steps)")
        else:
            h3.warning(f"🟡 Buffer Accumulating ({hw_seq_res['streak_length']}/24 steps)")

        if hw_seq_res["available"] and hw_seq_res["sequence_df"] is not None and not hw_seq_res["sequence_df"].empty:
            s_df = hw_seq_res["sequence_df"]
            start_str = (
                str(s_df["interval_start_utc"].iloc[0])[:19] + "Z"
                if "interval_start_utc" in s_df.columns and pd.notna(s_df["interval_start_utc"].iloc[0])
                else "N/A"
            )
            if "interval_end_utc" in s_df.columns and pd.notna(s_df["interval_end_utc"].iloc[-1]):
                end_str = str(s_df["interval_end_utc"].iloc[-1])[:19] + "Z"
            elif "interval_end_epoch" in s_df.columns and pd.notna(s_df["interval_end_epoch"].iloc[-1]):
                end_str = datetime.fromtimestamp(float(s_df["interval_end_epoch"].iloc[-1]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            elif "interval_start_utc" in s_df.columns and pd.notna(s_df["interval_start_utc"].iloc[-1]):
                end_str = (pd.to_datetime(s_df["interval_start_utc"].iloc[-1]) + pd.Timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            else:
                end_str = "N/A"

            p1_mean = f"{s_df['soil1_raw_mean'].mean():.1f}" if "soil1_raw_mean" in s_df.columns and pd.notna(s_df["soil1_raw_mean"].mean()) else "N/A"
            p2_mean = f"{s_df['soil2_raw_mean'].mean():.1f}" if "soil2_raw_mean" in s_df.columns and pd.notna(s_df["soil2_raw_mean"].mean()) else "N/A"
            temp_mean = f"{s_df['temperature_c_mean'].mean():.1f}°C" if "temperature_c_mean" in s_df.columns and pd.notna(s_df["temperature_c_mean"].mean()) else "N/A"

            st.info(
                f"ℹ️ **Current Active Hardware Sequence Window:** `{start_str}` to `{end_str}` • "
                f"Mean Probe 1 ADC: **{p1_mean}** • "
                f"Mean Probe 2 ADC: **{p2_mean}** • "
                f"Mean Air Temp: **{temp_mean}**\n\n"
                f"*Note: Live sequence inference remains blocked pending physical sensor additions and soil VWC calibration.*",
                icon="📡",
            )
        else:
            st.caption(f"Hardware buffer status: {hw_seq_res.get('blocker', 'No observation sequence available.')}")
    except Exception as e:
        st.error(f"Error auditing live sensor readiness: {e}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Agriculture Project • "
    "ESP32 + IoT + AI + Weather + Computer Vision"
)
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from datetime import datetime, timezone


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

DEFAULT_STATE = {
    "uploaded_image": None,
    "weed_detected": None,
    "weed_name": None,
    "weed_confidence": None,
    "recommendation": None,
    "last_detection_time": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SETTINGS
# ============================================================

# Real ESP32 / ThingSpeak data is NOT connected yet.
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


def get_sensor_data():
    """
    Use real ThingSpeak data when configured.
    Otherwise return N/A.
    """

    if (
        not USE_DEMO_DATA
        and THINGSPEAK_CHANNEL_ID
        and THINGSPEAK_READ_API_KEY
    ):
        sensor_data = get_thingspeak_data(
            THINGSPEAK_CHANNEL_ID,
            THINGSPEAK_READ_API_KEY,
        )

        if sensor_data is not None:
            return sensor_data

    return get_demo_sensor_data()


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
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"],
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
    model_name,
):
    """
    Irrigation AI is not integrated yet.

    Therefore we return N/A instead of fake logic.
    """

    return {
        "model": model_name,
        "required": None,
        "message": "N/A",
        "confidence": None,
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
    Historical ESP32/ThingSpeak data is not connected yet.
    """

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

model_name = st.sidebar.selectbox(
    "Irrigation AI Model",
    [
        "LSTM",
        "TCN",
        "Transformer",
    ],
)

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


st.sidebar.info(
    "ESP32 sensor data: N/A"
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

    c2.metric(
        "💧 Soil Moisture",
        display_value(
            sensor_data["soil_moisture"],
            " %",
        ),
    )

    c3.metric(
        "💦 Humidity",
        display_value(
            sensor_data["humidity"],
            " %",
        ),
    )

    c4.metric(
        "🪣 Water Level",
        display_value(
            sensor_data["water_level"],
            " %",
        ),
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
        s2.info("🪣 Water Level: N/A")
    elif sensor_data["water_level"] < 20:
        s2.error("🚨 Low Water Level")
    else:
        s2.success("✅ Water Available")

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

    st.info(
        f"Selected Model: {model_name}"
    )

    st.warning(
        "Irrigation AI: N/A"
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

    c1.metric(
        "Soil Moisture",
        display_value(
            sensor_data["soil_moisture"],
            " %",
        ),
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
        "Water Level",
        display_value(
            sensor_data["water_level"],
            " %",
        ),
    )

    st.divider()

    st.header(
        "🧠 AI Irrigation Prediction",
        anchor=False,
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Selected Model",
        model_name,
    )

    p2.metric(
        "Irrigation Decision",
        "N/A",
    )

    p3.metric(
        "Pump Status",
        display_value(
            sensor_data["pump_status"]
        ),
    )

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

    st.info(
        "Irrigation AI decision is N/A because "
        "the trained irrigation model has not "
        "been connected yet."
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
        "Deep-learning models for irrigation prediction."
    )

    st.divider()

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "LSTM",
                "TCN",
                "Transformer",
            ],
            "Approach": [
                "Recurrent",
                "Temporal Convolution",
                "Attention-based",
            ],
            "Training Status": [
                "N/A",
                "N/A",
                "N/A",
            ],
            "Testing Status": [
                "N/A",
                "N/A",
                "N/A",
            ],
            "RMSE": [
                "N/A",
                "N/A",
                "N/A",
            ],
            "MAE": [
                "N/A",
                "N/A",
                "N/A",
            ],
        }
    )

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Current Model"
    )

    st.info(
        f"Selected model: {model_name}"
    )

    st.write(
        "Model output: N/A — trained model "
        "is not connected yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Agriculture Project • "
    "ESP32 + IoT + AI + Weather + Computer Vision"
)
"""
Single source of truth for the preprocessing pipeline.

Every path, threshold and physical parameter used by the three pipeline stages is
defined here, so that a value can never be declared twice with two different
numbers. Import it as::

    import config as cfg

Stages
------
``00_irrigation_actuators_cleaning.py``  raw valve telemetry -> validated events
``01_merge_datasets.py``                 sensors + events + weather -> merged tables
``02_preprocessing.py``                  merged tables -> analysis-ready tables
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to the project root, i.e. the parent of this file's
# directory. Scripts can therefore be run from anywhere.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RAW_SENSOR_DIR = ROOT / "data" / "original" / "raw_sensor"
RAW_ACTUATOR_DIR = ROOT / "data" / "original" / "raw_irrigation_actuators"

EVENTS_DIR = ROOT / "data" / "irrigation_actuators"
MERGED_DIR = ROOT / "data" / "merged"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"

RAW_ACTUATORS_FILE = RAW_ACTUATOR_DIR / "raw_actuators.csv"
EVENTS_FILE = EVENTS_DIR / "final_irrigation_events.csv"
SECTOR_STATS_FILE = EVENTS_DIR / "sector_statistics_report.csv"

# Per-sensor-type raw exports consumed by stage 01. Each is a subset of
# MEASUREMENT_all.csv, split by sensor type.
SENSOR_FILES = {
    "ste": "MEASUREMENT_ste.csv",  # air temperature, outdoor
    "sti": "MEASUREMENT_sti.csv",  # air temperature, indoor
    "sue": "MEASUREMENT_sue.csv",  # air relative humidity, outdoor
    "sui": "MEASUREMENT_sui.csv",  # air relative humidity, indoor
    "ce": "MEASUREMENT_ce.csv",    # soil electrical conductivity, per sector
    "ph": "MEASUREMENT_ph.csv",    # soil pH, per sector
    "sut": "MEASUREMENT_sut.csv",  # soil moisture, per sector
}

# Sensor types shared by the whole farm, as opposed to those read per sector.
GLOBAL_SENSORS = ("ste", "sti", "sue", "sui")
ZONAL_SENSORS = ("ce", "ph", "sut")

# Raw sensor tag -> column name used from stage 01 onwards.
SENSOR_COLUMNS = {
    "ste": "outside_temperature",
    "sti": "inside_temperature",
    "sue": "outside_humidity",
    "sui": "inside_humidity",
    "ce": "ec",
    "ph": "ph",
    "sut": "soil_moisture",
}

# ---------------------------------------------------------------------------
# Site and sectors
# ---------------------------------------------------------------------------
LATITUDE = 40.349306   # Arnesano, Lecce, Apulia, Italy
LONGITUDE = 18.081556

SECTORS = (1, 2, 3, 4, 5)

# Nominal emitter flow rate per sector, litres per minute, used to convert valve
# opening time into applied litres. Sector 5 is served by a lower-capacity line
# than the others.
FLOW_RATES = {1: 1.1, 2: 1.1, 3: 1.1, 4: 1.1, 5: 0.4}

# Soil water-retention parameters, on the same 0-100 relative scale as the
# probe output. PWP permanent wilting point, MADP management allowed depletion
# point, FC field capacity, SP saturation point.
SOIL_TYPES = {
    "TOMATO_SOIL": {"PWP": 20.0, "MADP": 40.0, "FC": 60.0, "SP": 80.0},
    "TOMATO_POT": {"PWP": 30.0, "MADP": 60.0, "FC": 80.0, "SP": 90.0},
    "ZUCCHINI_SOIL": {"PWP": 40.0, "MADP": 80.0, "FC": 90.0, "SP": 95.0},
    "BLUEBERRY_SOIL": {"PWP": 35.0, "MADP": 70.0, "FC": 90.0, "SP": 95.0},
}

SECTOR_SOIL_TYPE = {
    1: "TOMATO_SOIL",
    2: "TOMATO_SOIL",
    3: "TOMATO_POT",
    4: "ZUCCHINI_SOIL",
    5: "BLUEBERRY_SOIL",
}

AGRONOMIC_CLASSES = {0: "Danger", 1: "Stress", 2: "Optimal", 3: "Excess", 4: "Saturation"}

# ---------------------------------------------------------------------------
# Stage 00 — irrigation event reconstruction
# ---------------------------------------------------------------------------
# Soil moisture below which irrigating is agronomically justified, per sector.
THRESHOLD_START = {1: 40, 2: 40, 3: 60, 4: 80, 5: 50}
# Soil moisture above which a standard irrigation is considered complete.
THRESHOLD_STOP = {1: 60, 2: 60, 3: 80, 4: 90, 5: 60}

FORBIDDEN_HOUR_START = 18  # no irrigation from 18:00 ...
FORBIDDEN_HOUR_END = 6     # ... until 06:00
FERTIGATION_TIMEOUT_MINUTES = 15
MERGE_TOLERANCE_MINUTES = 2   # events closer than this are one event
MIN_DURATION_MINUTES = 15     # shorter events are signal noise
MAX_DURATION_HOURS = 5        # longer events indicate a missing closure signal

# A long event is discarded only when it ended for one of these reasons; a long
# fertigation_timeout or humidity_threshold event is a genuine irrigation.
ANOMALOUS_STOP_REASONS = ("time_restriction", "physical_valve_closure")

# ---------------------------------------------------------------------------
# Stage 01 — merging
# ---------------------------------------------------------------------------
SAMPLING_INTERVAL = "10min"
SAMPLING_RATE_MINUTES = 10

WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo variable -> column name in the merged tables.
WEATHER_VARIABLES = {
    "temperature_2m": "weather_temp",
    "relative_humidity_2m": "weather_humidity",
    "rain": "weather_rain",
    "surface_pressure": "weather_pressure",
    "wind_speed_10m": "weather_wind_speed",
    "direct_radiation": "weather_radiation",
    "soil_temperature_0_to_7cm": "soil_temperature_0-7cm",
    # The API band is 7-28 cm; the column name is kept for continuity with the
    # published tables.
    "soil_temperature_7_to_28cm": "soil_temperature_7-18cm",
}

# ---------------------------------------------------------------------------
# Stage 02 — preprocessing
# ---------------------------------------------------------------------------
TARGET_HORIZON_HOURS = 24
PAST_WATER_WINDOW_HOURS = 4

# Minimum share of the 24 h window that must actually be sampled for the mean
# target to be computed. The probe reports intermittently — monthly coverage
# ranges from 3 % in February to 93 % in August — so requiring a fully sampled
# window would leave no usable row at all. Half the window is enough for the
# mean to describe the day rather than a single moment, while keeping the
# earlier part of the season. The point target has no equivalent tolerance: it
# either exists at t + 24 h or it does not.
MIN_WINDOW_COVERAGE = 0.5

# Columns dropped before saving: the on-site air sensors have too little
# coverage to be usable (73-94 % missing), and liters_total is superseded by the
# water_vol_* features.
COLS_TO_DROP = (
    "outside_temperature",
    "inside_temperature",
    "outside_humidity",
    "inside_humidity",
    "liters_total",
)

# Gaps of at most this many consecutive samples are interpolated; longer gaps
# are left missing and the affected rows are dropped.
INTERPOLATE_COLUMNS = ("soil_moisture", "ph", "ec")
INTERPOLATION_LIMIT = 2

# Soil moisture cleaning.
MOISTURE_VALID_RANGE = (0.0, 115.0)  # outside this range: sensor fault
MOISTURE_CLIP_MAX = 100.0            # 100-115 is over-saturation, clipped
SPIKE_WINDOW = 9                     # samples, centred
SPIKE_THRESHOLD = 15.0               # deviation from the rolling median
FLATLINE_WINDOW = 12                 # samples used to detect a stalled probe

# Plausibility bands for the remaining soil sensors. Readings outside them are
# error codes rather than measurements: the probes report values consistent with
# a 16-bit overflow when they lose contact or connection.
SENSOR_VALID_RANGES = {
    "ph": (3.0, 9.0),
    "ec": (0.0, 1000.0),
}


def agronomic_class(moisture, soil_type):
    """Map a soil moisture reading to its agronomic class (0-4).

    Returns None when the reading is missing.
    """
    if moisture is None or moisture != moisture:  # NaN-safe
        return None
    p = SOIL_TYPES[soil_type]
    if moisture < p["PWP"]:
        return 0
    if moisture < p["MADP"]:
        return 1
    if moisture < p["FC"]:
        return 2
    if moisture < p["SP"]:
        return 3
    return 4


def day_period(hour):
    """Coarse time-of-day bucket: 0 morning, 1 afternoon, 2 evening, 3 night."""
    if 5 <= hour < 12:
        return 0
    if 12 <= hour < 17:
        return 1
    if 17 <= hour < 21:
        return 2
    return 3


def flow_rate(sector):
    """Litres per minute delivered by the given sector's line."""
    return FLOW_RATES[sector]

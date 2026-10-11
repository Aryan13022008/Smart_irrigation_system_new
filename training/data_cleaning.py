from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

OUTPUT_FOLDER = PROJECT_ROOT / "training" / "processed"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SMART IRRIGATION - DATA CLEANING")
print("=" * 60)

print("\nLoading datasets...")

sensor_data = pd.read_excel(
    DATA_FOLDER / "sensordata2021_2022_2023.xlsx"
)

irrigation_data = pd.read_csv(
    DATA_FOLDER / "irrigation_212223.csv"
)

precipitation_data = pd.read_csv(
    DATA_FOLDER / "precipitation_212223.csv"
)

eto_data = pd.read_csv(
    DATA_FOLDER / "ETo_212223.csv"
)

soil_sample_data = pd.read_csv(
    DATA_FOLDER / "soilsampmeas_212223.csv"
)

sensor_summary = pd.read_excel(
    DATA_FOLDER / "Summary_sensors.xlsx"
)

print("All datasets loaded.")


# ============================================================
# 1. CLEAN SENSOR DATA
# ============================================================

print("\n" + "=" * 60)
print("CLEANING SENSOR DATA")
print("=" * 60)

sensor = sensor_data.copy()

# Convert datetime
sensor["Datetime"] = pd.to_datetime(
    sensor["Datetime"],
    errors="coerce"
)

# Remove rows where datetime is invalid
sensor = sensor.dropna(
    subset=["Datetime"]
)

# Sort chronologically
sensor = sensor.sort_values(
    ["Sensor", "Datetime"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Soil moisture
# ------------------------------------------------------------

vwc_columns = [
    "vwc0 (m3/m3)",
    "vwc1 (m3/m3)",
    "vwc2 (m3/m3)",
]

for column in vwc_columns:

    sensor[column] = pd.to_numeric(
        sensor[column],
        errors="coerce"
    )

    # Invalid VWC becomes missing
    sensor.loc[
        (sensor[column] < 0.01) |
        (sensor[column] > 1.00),
        column
    ] = pd.NA


# Average the three soil-moisture sensors
sensor["soil_moisture"] = sensor[
    vwc_columns
].mean(axis=1)


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

sensor["temp"] = pd.to_numeric(
    sensor["temp"],
    errors="coerce"
)

# Physically unreasonable values become missing
sensor.loc[
    (sensor["temp"] < -20) |
    (sensor["temp"] > 60),
    "temp"
] = pd.NA


# ------------------------------------------------------------
# Rain gauge
# ------------------------------------------------------------

sensor["pluvio"] = pd.to_numeric(
    sensor["pluvio"],
    errors="coerce"
)

sensor.loc[
    sensor["pluvio"] < 0,
    "pluvio"
] = pd.NA


# ============================================================
# 2. CREATE DAILY SENSOR DATA
# ============================================================

print("\nCreating daily sensor data...")

sensor["Date"] = sensor["Datetime"].dt.normalize()

daily_sensor = (
    sensor
    .groupby(["Date", "Sensor"])
    .agg(
        soil_moisture=("soil_moisture", "mean"),
        temperature=("temp", "mean"),
        rainfall=("pluvio", "sum"),
    )
    .reset_index()
)

print(
    "Daily sensor rows:",
    len(daily_sensor)
)


# ============================================================
# 3. CLEAN IRRIGATION DATA
# ============================================================

print("\nCleaning irrigation data...")

irrigation = irrigation_data.copy()

irrigation["Date"] = pd.to_datetime(
    irrigation["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

irrigation = irrigation.dropna(
    subset=["Date"]
)

# Convert sensor columns to numbers
irrigation_sensor_columns = [
    column
    for column in irrigation.columns
    if column not in ["year", "Date"]
]

for column in irrigation_sensor_columns:

    irrigation[column] = pd.to_numeric(
        irrigation[column],
        errors="coerce"
    )


# ============================================================
# 4. CONVERT IRRIGATION DATA TO LONG FORMAT
# ============================================================

print("Preparing irrigation data...")

irrigation_long = irrigation.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="irrigation"
)

irrigation_long["Date"] = (
    irrigation_long["Date"].dt.normalize()
)

# Missing irrigation values are left missing.
# Zero means the dataset actually recorded zero irrigation.

irrigation_long = irrigation_long[
    ["Date", "Sensor", "irrigation"]
]


# ============================================================
# 5. CLEAN PRECIPITATION DATA
# ============================================================

print("\nCleaning precipitation data...")

precipitation = precipitation_data.copy()

precipitation["Date"] = pd.to_datetime(
    precipitation["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

precipitation = precipitation.dropna(
    subset=["Date"]
)

precipitation_sensor_columns = [
    column
    for column in precipitation.columns
    if column not in ["year", "Date"]
]

for column in precipitation_sensor_columns:

    precipitation[column] = pd.to_numeric(
        precipitation[column],
        errors="coerce"
    )


# Convert wide → long
precipitation_long = precipitation.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="precipitation"
)

precipitation_long["Date"] = (
    precipitation_long["Date"].dt.normalize()
)

precipitation_long = precipitation_long[
    ["Date", "Sensor", "precipitation"]
]


# ============================================================
# 6. CLEAN ETo DATA
# ============================================================

print("\nCleaning ETo data...")

eto = eto_data.copy()

eto["Date"] = pd.to_datetime(
    eto["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

eto = eto.dropna(
    subset=["Date"]
)

eto_sensor_columns = [
    column
    for column in eto.columns
    if column not in ["year", "Date"]
]

for column in eto_sensor_columns:

    eto[column] = pd.to_numeric(
        eto[column],
        errors="coerce"
    )


# Wide → long
eto_long = eto.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="eto"
)

eto_long["Date"] = (
    eto_long["Date"].dt.normalize()
)

eto_long = eto_long[
    ["Date", "Sensor", "eto"]
]


# ============================================================
# 7. CLEAN SOIL SAMPLE DATA
# ============================================================

print("\nCleaning soil sample data...")

soil = soil_sample_data.copy()

soil["Date"] = pd.to_datetime(
    soil["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

soil = soil.dropna(
    subset=["Date"]
)

soil["Date"] = soil["Date"].dt.normalize()


# ============================================================
# 8. COMBINE SENSOR + WEATHER + IRRIGATION
# ============================================================

print("\nCombining datasets...")

combined = daily_sensor.merge(
    irrigation_long,
    on=["Date", "Sensor"],
    how="left"
)

combined = combined.merge(
    precipitation_long,
    on=["Date", "Sensor"],
    how="left"
)

combined = combined.merge(
    eto_long,
    on=["Date", "Sensor"],
    how="left"
)


# ============================================================
# 9. ADD USEFUL FEATURES
# ============================================================

print("\nCreating ML features...")

combined = combined.sort_values(
    ["Sensor", "Date"]
).reset_index(drop=True)


# Previous day's soil moisture
combined["soil_moisture_previous"] = (
    combined
    .groupby("Sensor")["soil_moisture"]
    .shift(1)
)


# Soil moisture change
combined["soil_moisture_change"] = (
    combined["soil_moisture"]
    - combined["soil_moisture_previous"]
)


# Previous rainfall
combined["rainfall_previous"] = (
    combined
    .groupby("Sensor")["precipitation"]
    .shift(1)
)


# Previous ETo
combined["eto_previous"] = (
    combined
    .groupby("Sensor")["eto"]
    .shift(1)
)


# ============================================================
# 10. CREATE IRRIGATION TARGET
# ============================================================

print("\nCreating irrigation target...")

# For now:
# irrigation > 0 means irrigation occurred.
#
# This is NOT the final AI target.
# We are creating a first target so we can inspect it.

combined["irrigation_target"] = (
    combined["irrigation"]
    .fillna(0)
    > 0
).astype(int)


# ============================================================
# 11. SAVE CLEAN DATA
# ============================================================

output_file = (
    OUTPUT_FOLDER /
    "clean_irrigation_dataset.csv"
)

combined.to_csv(
    output_file,
    index=False
)


# ============================================================
# 12. REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print("\nOriginal sensor rows:")
print(len(sensor_data))

print("\nDaily sensor rows:")
print(len(daily_sensor))

print("\nFinal combined rows:")
print(len(combined))

print("\nFinal columns:")
print(list(combined.columns))

print("\nMissing values in final dataset:")

missing = combined.isna().sum()

print(
    missing[missing > 0]
    if missing.sum() > 0
    else "No missing values"
)

print("\nIrrigation target distribution:")

print(
    combined["irrigation_target"]
    .value_counts()
)

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 60)
print("NEXT STEP: INSPECT THE CLEAN DATASET")
print("=" * 60)
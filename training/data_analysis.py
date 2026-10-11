from pathlib import Path
import pandas as pd

# ============================================================
# SMART IRRIGATION - DATASET ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

print("\n" + "=" * 60)
print("SMART IRRIGATION DATASET REPORT")
print("=" * 60)

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

required_files = [
    "sensordata2021_2022_2023.xlsx",
    "irrigation_212223.csv",
    "precipitation_212223.csv",
    "ETo_212223.csv",
    "soilsampmeas_212223.csv",
    "Summary_sensors.xlsx",
]

print("\n--- CHECKING DATA FILES ---")

for filename in required_files:
    path = DATA_FOLDER / filename

    if path.exists():
        print(f"[OK] {filename}")
    else:
        print(f"[MISSING] {filename}")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\n--- LOADING DATA ---")

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

print("[OK] All datasets loaded successfully.")

# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

datasets = {
    "SENSOR DATA": sensor_data,
    "IRRIGATION DATA": irrigation_data,
    "PRECIPITATION DATA": precipitation_data,
    "ETo DATA": eto_data,
    "SOIL SAMPLE DATA": soil_sample_data,
    "SENSOR SUMMARY": sensor_summary,
}

for name, df in datasets.items():

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(list(df.columns))

# ------------------------------------------------------------
# Sensor data
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SENSOR DATA INSPECTION")
print("=" * 60)

print("\nFirst 5 rows:")
print(sensor_data.head())

print("\nData types:")
print(sensor_data.dtypes)

# ------------------------------------------------------------
# Date information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATE INFORMATION")
print("=" * 60)

sensor_data["Datetime"] = pd.to_datetime(
    sensor_data["Datetime"],
    errors="coerce"
)

print("\nSensor data date range:")

print(
    "Start:",
    sensor_data["Datetime"].min()
)

print(
    "End:",
    sensor_data["Datetime"].max()
)

# ------------------------------------------------------------
# Sensor modules
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SENSOR MODULES")
print("=" * 60)

print(
    "\nNumber of unique sensors:",
    sensor_data["Sensor"].nunique()
)

print("\nSensor IDs:")
print(sensor_data["Sensor"].unique())

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

for name, df in datasets.items():

    missing = df.isnull().sum()

    print(f"\n{name}:")

    print(
        missing[missing > 0]
        if missing.sum() > 0
        else "No missing values"
    )

# ------------------------------------------------------------
# Soil moisture inspection
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SOIL MOISTURE CHECK")
print("=" * 60)

vwc_columns = [
    "vwc0 (m3/m3)",
    "vwc1 (m3/m3)",
    "vwc2 (m3/m3)",
]

for column in vwc_columns:

    if column in sensor_data.columns:

        values = sensor_data[column].dropna()

        print(f"\n{column}")

        print("Minimum:", values.min())
        print("Maximum:", values.max())
        print("Mean:", values.mean())

        invalid = ((values < 0.01) | (values > 1.00)).sum()

        print(
            "Values outside expected range:",
            invalid
        )

# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEMPERATURE CHECK")
print("=" * 60)

if "temp" in sensor_data.columns:

    temperature = sensor_data["temp"].dropna()

    print("Minimum:", temperature.min())
    print("Maximum:", temperature.max())
    print("Mean:", temperature.mean())

# ------------------------------------------------------------
# Precipitation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PRECIPITATION DATA CHECK")
print("=" * 60)

print("\nPrecipitation dataset:")
print(precipitation_data.head())

# ------------------------------------------------------------
# Irrigation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("IRRIGATION DATA CHECK")
print("=" * 60)

print("\nIrrigation dataset:")
print(irrigation_data.head())

# ------------------------------------------------------------
# ETo
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVAPOTRANSPIRATION (ETo) CHECK")
print("=" * 60)

print("\nETo dataset:")
print(eto_data.head())

# ------------------------------------------------------------
# Soil samples
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SOIL SAMPLE CHECK")
print("=" * 60)

print("\nSoil sample dataset:")
print(soil_sample_data.head())

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETE")
print("=" * 60)

print(
    "\nNext stage:"
    "\n1. Clean and synchronize the datasets"
    "\n2. Create useful ML features"
    "\n3. Define the irrigation prediction target"
    "\n4. Train and evaluate the AI model"
)

print("\nNo AI model has been trained yet.")
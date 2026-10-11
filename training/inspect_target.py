from pathlib import Path
import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "training"
    / "processed"
    / "clean_irrigation_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("IRRIGATION TARGET INSPECTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# FIX DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# IRRIGATION TARGET
# ============================================================

print("\nIrrigation target:")
print(
    df["irrigation_target"].value_counts()
)

print("\nPercentage:")
print(
    df["irrigation_target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# CONDITIONS DURING IRRIGATION
# ============================================================

print("\n" + "=" * 60)
print("CONDITIONS DURING IRRIGATION")
print("=" * 60)

irrigated = df[
    df["irrigation_target"] == 1
].copy()

columns = [
    "soil_moisture",
    "temperature",
    "precipitation",
    "eto",
    "soil_moisture_change",
]

print("\nNumber of irrigation records:")
print(len(irrigated))

print("\nAverage values:")
print(
    irrigated[columns]
    .mean(numeric_only=True)
)

print("\nMinimum values:")
print(
    irrigated[columns]
    .min(numeric_only=True)
)

print("\nMaximum values:")
print(
    irrigated[columns]
    .max(numeric_only=True)
)


# ============================================================
# CONDITIONS WHEN IRRIGATION DID NOT OCCUR
# ============================================================

print("\n" + "=" * 60)
print("CONDITIONS WHEN IRRIGATION DID NOT OCCUR")
print("=" * 60)

not_irrigated = df[
    df["irrigation_target"] == 0
].copy()

print("\nNumber of non-irrigation records:")
print(len(not_irrigated))

print("\nAverage values:")
print(
    not_irrigated[columns]
    .mean(numeric_only=True)
)


# ============================================================
# IRRIGATION EVENTS BY SENSOR
# ============================================================

print("\n" + "=" * 60)
print("IRRIGATION EVENTS BY SENSOR")
print("=" * 60)

sensor_events = (
    irrigated["Sensor"]
    .value_counts()
)

print(sensor_events)


# ============================================================
# IRRIGATION EVENTS BY YEAR
# ============================================================

print("\n" + "=" * 60)
print("IRRIGATION EVENTS BY YEAR")
print("=" * 60)

# Date has already been converted above
irrigation_by_year = (
    irrigated
    .dropna(subset=["Date"])
    .groupby(
        irrigated.dropna(subset=["Date"])["Date"].dt.year
    )
    .size()
)

print(irrigation_by_year)


# ============================================================
# IRRIGATION EVENTS BY MONTH
# ============================================================

print("\n" + "=" * 60)
print("IRRIGATION EVENTS BY MONTH")
print("=" * 60)

irrigation_by_month = (
    irrigated
    .dropna(subset=["Date"])
    .groupby(
        irrigated.dropna(subset=["Date"])["Date"].dt.month
    )
    .size()
)

print(irrigation_by_month)


# ============================================================
# SOIL MOISTURE DURING IRRIGATION
# ============================================================

print("\n" + "=" * 60)
print("SOIL MOISTURE DURING IRRIGATION")
print("=" * 60)

soil_values = irrigated[
    "soil_moisture"
].dropna()

if len(soil_values) > 0:

    print(
        f"\nAverage: {soil_values.mean():.3f}"
    )

    print(
        f"Minimum: {soil_values.min():.3f}"
    )

    print(
        f"Maximum: {soil_values.max():.3f}"
    )

else:

    print("\nNo valid soil moisture values available.")


# ============================================================
# DATA QUALITY FOR TARGET
# ============================================================

print("\n" + "=" * 60)
print("TARGET DATA QUALITY")
print("=" * 60)

print(
    "\nMissing irrigation targets:"
)

print(
    df["irrigation_target"]
    .isna()
    .sum()
)

print(
    "\nValid dates:"
)

print(
    df["Date"]
    .notna()
    .sum()
)

print(
    "\nInvalid/missing dates:"
)

print(
    df["Date"]
    .isna()
    .sum()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

print(
    """
The dataset has been successfully inspected.

Important findings will now be used to:
1. Select the final ML features
2. Handle missing values correctly
3. Deal with class imbalance
4. Build the training dataset
5. Train the first irrigation prediction model
"""
)
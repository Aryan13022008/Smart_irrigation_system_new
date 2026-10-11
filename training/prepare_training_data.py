from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "training"
    / "processed"
    / "clean_irrigation_dataset.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "training"
    / "processed"
)

OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# LOAD CLEAN DATA
# ============================================================

print("=" * 60)
print("SMART IRRIGATION - TRAINING DATA PREPARATION")
print("=" * 60)

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows.")


# ============================================================
# DATE PROCESSING
# ============================================================

print("\nProcessing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
).copy()

# Calendar features
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day_of_year"] = df["Date"].dt.dayofyear

# Cyclic representation of the year.
# This helps the model understand that December and January
# are close to each other.
import numpy as np

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# ============================================================
# SENSOR ENCODING
# ============================================================

print("Encoding sensor information...")

# One-hot encoding prevents the model from treating sensor IDs
# as numerical quantities.
sensor_encoded = pd.get_dummies(
    df["Sensor"],
    prefix="sensor",
    dtype=int
)

df = pd.concat(
    [df, sensor_encoded],
    axis=1
)


# ============================================================
# SELECT FEATURES
# ============================================================

print("\nSelecting ML features...")

base_features = [
    "soil_moisture",
    "temperature",
    "precipitation",
    "eto",
    "soil_moisture_previous",
    "soil_moisture_change",
    "rainfall_previous",
    "eto_previous",
    "month_sin",
    "month_cos",
]

sensor_features = list(
    sensor_encoded.columns
)

feature_columns = (
    base_features +
    sensor_features
)


# ============================================================
# CHECK FEATURE AVAILABILITY
# ============================================================

print("\nChecking missing feature values...")

print(
    df[feature_columns]
    .isna()
    .sum()
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

# Important:
# We use median values calculated from the training portion
# later during model training.
#
# For this preparation stage we remove rows where the main
# sensor measurement is unavailable because soil moisture is
# one of the most important inputs.

df = df.dropna(
    subset=["soil_moisture"]
).copy()

print(
    f"Rows after removing missing soil moisture: {len(df)}"
)


# ============================================================
# TARGET
# ============================================================

print("\nPreparing irrigation target...")

target_column = "irrigation_target"

print("\nTarget distribution:")

print(
    df[target_column]
    .value_counts()
)

print("\nTarget percentage:")

print(
    df[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting chronologically...")

df = df.sort_values(
    ["Date", "Sensor"]
).reset_index(drop=True)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

# We deliberately DO NOT randomly shuffle the data.
#
# The model should learn from the past and be tested on
# later dates, which better represents real-world operation.

split_date = pd.Timestamp("2023-01-01")

train_df = df[
    df["Date"] < split_date
].copy()

test_df = df[
    df["Date"] >= split_date
].copy()


# ============================================================
# CREATE X AND Y
# ============================================================

X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    target_column
].copy()

X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    target_column
].copy()


# ============================================================
# HANDLE REMAINING MISSING VALUES
# ============================================================

print("\nFilling remaining feature gaps...")

# Calculate medians ONLY from training data.
# This avoids information leakage from the test period.

train_medians = X_train.median(
    numeric_only=True
)

X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving prepared datasets...")

X_train.to_csv(
    OUTPUT_FOLDER / "X_train.csv",
    index=False
)

X_test.to_csv(
    OUTPUT_FOLDER / "X_test.csv",
    index=False
)

y_train.to_csv(
    OUTPUT_FOLDER / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_FOLDER / "y_test.csv",
    index=False
)


# Also save the complete processed dataset
df.to_csv(
    OUTPUT_FOLDER / "final_training_dataset.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nTraining period:")
print(
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)

print("\nTesting period:")
print(
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)

print("\nTraining rows:")
print(len(X_train))

print("\nTesting rows:")
print(len(X_test))

print("\nNumber of features:")
print(len(feature_columns))

print("\nTraining target:")
print(
    y_train.value_counts()
)

print("\nTesting target:")
print(
    y_test.value_counts()
)

print("\nFeature columns:")

for feature in feature_columns:
    print(
        " -",
        feature
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\nMissing values in X_train:")
print(
    X_train.isna().sum().sum()
)

print("\nMissing values in X_test:")
print(
    X_test.isna().sum().sum()
)

print("\nFiles created:")

print(
    OUTPUT_FOLDER / "X_train.csv"
)

print(
    OUTPUT_FOLDER / "X_test.csv"
)

print(
    OUTPUT_FOLDER / "y_train.csv"
)

print(
    OUTPUT_FOLDER / "y_test.csv"
)

print(
    OUTPUT_FOLDER / "final_training_dataset.csv"
)

print("\n" + "=" * 60)
print("READY FOR MODEL TRAINING")
print("=" * 60)
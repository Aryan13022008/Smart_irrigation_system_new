"""
Shared utilities for the sequence-based models (LSTM, TCN, Transformer).
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "training" / "processed"

# Time-varying signals (matches real IoT dataset)
TIME_FEATURES = [
    "soil_moisture",
    "soil_temperature", 
    "soil_humidity", 
    "temperature",
    "wind_speed", 
    "air_humidity", 
    "wind_gust", 
    "pressure",
    "soil_moisture_previous", 
    "soil_moisture_change", 
    "temp_previous",
    "month_sin", 
    "month_cos"
]

TARGET_COL = "irrigation_target"
SENSOR_COL = "Sensor"
DATE_COL = "Date"

# Changed from 7 to 72. Because data is hourly, 72 steps = 3 full days of historical context
WINDOW_SIZE = 24  


def load_final_dataset():
    df = pd.read_csv(PROCESSED_DIR / "final_training_dataset.csv")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([SENSOR_COL, DATE_COL]).reset_index(drop=True)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    return df, sensor_cols


def fill_missing_time_features(df):
    df = df.copy()
    df[TIME_FEATURES] = df.groupby(SENSOR_COL)[TIME_FEATURES].transform(
        lambda group: group.ffill().bfill()
    )
    fallback_medians = df[TIME_FEATURES].median(numeric_only=True)
    df[TIME_FEATURES] = df[TIME_FEATURES].fillna(fallback_medians)
    return df


def build_sequences(df, sensor_cols, window_size=WINDOW_SIZE):
    feature_cols = TIME_FEATURES + sensor_cols
    X_list, y_list, date_list = [], [], []

    for sensor, group in df.groupby(SENSOR_COL):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[TARGET_COL].to_numpy(dtype=np.float32)
        dates = group[DATE_COL].to_numpy()

        for end_idx in range(window_size - 1, len(group) - 1):
            start_idx = end_idx - window_size + 1
            window = feats[start_idx : end_idx + 1]
            label = targets[end_idx + 1]
            label_date = dates[end_idx + 1]

            X_list.append(window)
            y_list.append(label)
            date_list.append(label_date)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)

    return X, y, dates, feature_cols


def chronological_split(X, y, dates):
    # Dynamically find the 75% cutoff date instead of hardcoding a timestamp
    sorted_unique_dates = np.sort(np.unique(dates))
    split_idx = int(len(sorted_unique_dates) * 0.75)
    split_date = sorted_unique_dates[split_idx]
    
    train_mask = dates < split_date
    test_mask = ~train_mask

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def chronological_train_val_split(X_train, y_train, train_dates, val_fraction=0.2):
    order = np.argsort(train_dates)
    sorted_dates = train_dates[order]
    n = len(sorted_dates)
    n_val = max(1, int(round(n * val_fraction)))
    cutoff_date = sorted_dates[-n_val]

    val_mask = train_dates >= cutoff_date
    fit_mask = ~val_mask

    return X_train[fit_mask], X_train[val_mask], y_train[fit_mask], y_train[val_mask]


def scale_features(X_train, X_test, time_feature_count):
    train_flat = X_train[:, :, :time_feature_count].reshape(-1, time_feature_count)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std == 0] = 1.0

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[:, :, :time_feature_count] = (X_train[:, :, :time_feature_count] - mean) / std
    X_test[:, :, :time_feature_count] = (X_test[:, :, :time_feature_count] - mean) / std

    return X_train, X_test, mean, std


def get_train_test_sequences(window_size=WINDOW_SIZE, verbose=True):
    df, sensor_cols = load_final_dataset()
    df = fill_missing_time_features(df)
    X, y, dates, feature_cols = build_sequences(df, sensor_cols, window_size)
    X_train, X_test, y_train, y_test = chronological_split(X, y, dates)
    X_train, X_test, mean, std = scale_features(X_train, X_test, time_feature_count=len(TIME_FEATURES))

    if verbose:
        print(f"Window size        : {window_size} hours")
        print(f"Total sequences    : {len(X)}")
        print(f"Train sequences    : {len(X_train)}  (positives: {int(y_train.sum())})")
        print(f"Test sequences     : {len(X_test)}  (positives: {int(y_test.sum())})")

    return X_train, X_test, y_train, y_test, feature_cols


def get_train_val_test_sequences(window_size=WINDOW_SIZE, val_fraction=0.2, verbose=True):
    df, sensor_cols = load_final_dataset()
    df = fill_missing_time_features(df)
    X, y, dates, feature_cols = build_sequences(df, sensor_cols, window_size)
    
    # 75/25 split
    X_train, X_test, y_train, y_test = chronological_split(X, y, dates)
    
    sorted_unique_dates = np.sort(np.unique(dates))
    split_date = sorted_unique_dates[int(len(sorted_unique_dates) * 0.75)]
    train_dates = dates[dates < split_date]

    X_fit, X_val, y_fit, y_val = chronological_train_val_split(
        X_train, y_train, train_dates, val_fraction=val_fraction
    )

    time_feature_count = len(TIME_FEATURES)
    train_flat = X_fit[:, :, :time_feature_count].reshape(-1, time_feature_count)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std == 0] = 1.0

    def apply_scale(arr):
        arr = arr.copy()
        arr[:, :, :time_feature_count] = (arr[:, :, :time_feature_count] - mean) / std
        return arr

    X_fit, X_val, X_test = apply_scale(X_fit), apply_scale(X_val), apply_scale(X_test)

    if verbose:
        print(f"Window size        : {window_size} hours")
        print(f"Total sequences    : {len(X)}")
        print(f"Fit sequences      : {len(X_fit)}  (positives: {int(y_fit.sum())})")
        print(f"Val sequences      : {len(X_val)}  (positives: {int(y_val.sum())})")
        print(f"Test sequences     : {len(X_test)}  (positives: {int(y_test.sum())})")

    return X_fit, X_val, X_test, y_fit, y_val, y_test, feature_cols

"""
Smart Agriculture - 11-Feature Reduced Model Training Pipeline
=============================================================
Trains a dedicated HistGradientBoostingClassifier using only legitimately
available sensors (excluding physical soil_temperature and soil_humidity).

Preserves the original 13-feature baseline (random_forest.pkl) and records
metrics separately in random_forest_11feat_metrics.json.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "training"))
from threshold_utils import chronological_val_split, tune_threshold

# Paths
MODELS_DIR = PROJECT_ROOT / "training" / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "random_forest_11feat.pkl"
METRICS_OUTPUT_PATH = MODELS_DIR / "random_forest_11feat_metrics.json"

RAW_DATA_PATHS = [
    PROJECT_ROOT / "data" / "iotsensordata.csv",
    Path("/Users/aryanagrawal/Downloads/iotsensordata.csv"),
]

# Exact 11-Feature Schema (ordered)
ORDERED_11_FEATURES = [
    "soil_moisture",
    "temperature",
    "wind_speed",
    "air_humidity",
    "wind_gust",
    "pressure",
    "soil_moisture_previous",
    "soil_moisture_change",
    "temp_previous",
    "month_sin",
    "month_cos",
]


def load_and_preprocess_data():
    """Load and preprocess iotsensordata.csv with exact chronological integrity."""
    data_file = None
    for p in RAW_DATA_PATHS:
        if p.exists():
            data_file = p
            break

    if data_file is None:
        raise FileNotFoundError(
            f"Dataset iotsensordata.csv not found in search paths: {[str(p) for p in RAW_DATA_PATHS]}"
        )

    print(f"Loading raw dataset from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"Initial raw rows: {len(df)}")

    # Guard against target leakage
    banned_leakage_cols = ["ph", "rainfall", "N", "P", "K", "Time"]
    df = df.drop(columns=[c for c in banned_leakage_cols if c in df.columns])

    # Clean rows with missing values across required columns
    df = df.dropna().reset_index(drop=True)
    print(f"Cleaned rows after dropna: {len(df)}")

    # Column mapping
    rename_map = {
        "Soil Moisture": "soil_moisture",
        "Temperature": "soil_temperature_raw_discarded",  # Excluded
        " Soil Humidity": "soil_humidity_raw_discarded",   # Excluded
        "Air temperature (C)": "temperature",
        "Wind speed (Km/h)": "wind_speed",
        "Air humidity (%)": "air_humidity",
        "Wind gust (Km/h)": "wind_gust",
        "Pressure (KPa)": "pressure",
    }
    df = df.rename(columns=rename_map)

    # Target variable
    df["irrigation_target"] = df["Status"].map({"ON": 1, "OFF": 0})
    if df["irrigation_target"].isna().any():
        raise ValueError("Invalid target values encountered in Status column.")

    # Chronological timestamps (1 hour intervals)
    df["Date"] = pd.date_range(start="2022-01-01", periods=len(df), freq="1h")

    # Lagged feature construction (strictly backwards looking)
    df["soil_moisture_previous"] = df["soil_moisture"].shift(1)
    df["soil_moisture_change"] = df["soil_moisture"] - df["soil_moisture_previous"]
    df["temp_previous"] = df["temperature"].shift(1)

    # Cyclical astronomical clock features
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

    # Fill initial boundary row 0
    df = df.bfill()

    return df, data_file


def compute_metrics(y_true, y_pred, y_proba):
    """Compute standardized classification evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def calculate_sample_weights(y):
    """Calculate balanced sample weights for binary classification."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * max(n_pos, 1))
    w_neg = len(y) / (2 * max(n_neg, 1))
    return y.map({0: w_neg, 1: w_pos}).to_numpy()


def train_and_evaluate():
    """Main execution function."""
    print("=" * 70)
    print("SMART AGRICULTURE — TRAINING 11-FEATURE REDUCED MODEL")
    print("=" * 70)

    df, data_file = load_and_preprocess_data()

    # Chronological split: 75% train, 25% test
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nTrain set: {len(train_df)} rows ({train_df['irrigation_target'].sum()} positive)")
    print(f"Test set:  {len(test_df)} rows ({test_df['irrigation_target'].sum()} positive)")

    X_train = train_df[ORDERED_11_FEATURES]
    y_train = train_df["irrigation_target"]
    X_test = test_df[ORDERED_11_FEATURES]
    y_test = test_df["irrigation_target"]

    # Chronological validation split strictly within training period
    fold_train_mask, val_mask = chronological_val_split(
        np.arange(len(X_train)), val_fraction=0.2
    )
    X_fit, X_val = X_train.iloc[fold_train_mask], X_train.iloc[val_mask]
    y_fit, y_val = y_train.iloc[fold_train_mask], y_train.iloc[val_mask]

    print(
        f"\nInternal chronological validation split (from training period only):\n"
        f"  Fit: {len(X_fit)} rows ({int(y_fit.sum())} positive)\n"
        f"  Val: {len(X_val)} rows ({int(y_val.sum())} positive)"
    )

    # 1. Fit on fit-split to tune threshold on validation set
    clf_val = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    clf_val.fit(X_fit, y_fit, sample_weight=calculate_sample_weights(y_fit))

    val_proba = clf_val.predict_proba(X_val)[:, 1]
    tuned_threshold, val_stats = tune_threshold(y_val, val_proba, strategy="f1")
    print(f"\nTuned decision threshold (from validation set): {tuned_threshold:.4f}")
    print(f"  Val Precision: {val_stats['precision']:.4f}, Recall: {val_stats['recall']:.4f}, F1: {val_stats['f1']:.4f}")

    # 2. Refit final model on the FULL training split
    clf_full = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    clf_full.fit(X_train, y_train, sample_weight=calculate_sample_weights(y_train))

    # 3. Evaluate on held-out test set
    test_proba = clf_full.predict_proba(X_test)[:, 1]

    # Metrics at default 0.50 threshold
    metrics_default = compute_metrics(y_test, (test_proba >= 0.50).astype(int), test_proba)

    # Metrics at tuned threshold
    metrics_tuned = compute_metrics(y_test, (test_proba >= tuned_threshold).astype(int), test_proba)

    print("\n" + "=" * 60)
    print("TEST METRICS @ DEFAULT THRESHOLD (0.50)")
    print("=" * 60)
    for k, v in metrics_default.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v:.4f}")
    print("  confusion_matrix:", metrics_default["confusion_matrix"])

    print("\n" + "=" * 60)
    print(f"TEST METRICS @ TUNED THRESHOLD ({tuned_threshold:.4f})")
    print("=" * 60)
    for k, v in metrics_tuned.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v:.4f}")
    print("  confusion_matrix:", metrics_tuned["confusion_matrix"])

    # 4. Save model artifact
    joblib.dump(clf_full, MODEL_OUTPUT_PATH)
    print(f"\nSaved 11-feature model artifact to: {MODEL_OUTPUT_PATH}")

    # Feature medians from training split (for documentation)
    feature_medians = X_train.median(numeric_only=True).to_dict()

    # 5. Save comprehensive metrics file
    metrics_payload = {
        "model_name": "random_forest_11feat",
        "model_type": "hist_gradient_boosting",
        "estimator_class": clf_full.__class__.__name__,
        "dataset_source": str(data_file),
        "total_dataset_rows": len(df),
        "split_boundaries": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_fraction": 0.75,
            "test_fraction": 0.25,
            "chronological": True,
        },
        "features": ORDERED_11_FEATURES,
        "n_features": len(ORDERED_11_FEATURES),
        "excluded_features": ["soil_temperature", "soil_humidity"],
        "feature_medians": feature_medians,
        "threshold_default": 0.50,
        "threshold_tuned": float(tuned_threshold),
        "validation_stats": val_stats,
        "test_at_default_threshold": metrics_default,
        "test_at_tuned_threshold": metrics_tuned,
        "physical_soil_mapping_validated": False,
        "deployment_status": "experimental_offline_only",
    }

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved 11-feature metrics JSON to: {METRICS_OUTPUT_PATH}")

    return clf_full, metrics_payload


if __name__ == "__main__":
    train_and_evaluate()

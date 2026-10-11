"""
Smart Irrigation System - Baseline Model Training
==================================================

Trains classifiers to predict irrigation_target (0 = no irrigation,
1 = irrigation) using the prepared, chronologically-split feature set.

v2 changes vs. the original baseline script
--------------------------------------------
The first version trained a single RandomForestClassifier(class_weight
="balanced") and reported metrics at the default 0.5 threshold. On
this dataset that produced:

    Accuracy 0.95 / Precision 1.00 / Recall 0.02 / ROC-AUC 0.83

i.e. a model that is nearly useless operationally (it almost never
flags irrigation) despite looking "good" on accuracy and even
precision. ROC-AUC 0.83 / PR-AUC 0.23 show the model's *ranking* of
risky vs. safe days is actually reasonable — 0.5 is just the wrong
cutoff for a ~6% positive rate.

This version:
  1. Carves a chronological VALIDATION split out of the training
     period only (never touches the test set) to do model selection
     and threshold tuning without leaking test information.
  2. Compares a small set of candidate models/hyperparameters on
     validation PR-AUC (the right metric for rare positives).
  3. Refits the best candidate on the full training period.
  4. Tunes the decision threshold on the validation set (F1-optimal),
     then reports test metrics at BOTH 0.5 and the tuned threshold so
     the improvement from thresholding alone is visible.
  5. Saves the model, the tuned threshold, and a metrics JSON that the
     dashboard reads to populate the "AI Models" comparison table.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from threshold_utils import chronological_val_split, tune_threshold

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "training" / "processed"
MODELS_DIR = PROJECT_ROOT / "training" / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def leakage_check(feature_columns):
    """
    Guard against the raw `irrigation` column (or anything that
    directly encodes the action being predicted) sneaking into
    the feature set.
    """
    banned = {"irrigation", "irrigation_target"}
    leaked = banned.intersection(set(feature_columns))
    if leaked:
        raise ValueError(
            f"Data leakage detected — banned column(s) present in "
            f"features: {leaked}"
        )
    print("Leakage check passed: no target-derived columns in features.")


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(title, m):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}  <-- most important for this project")
    print(f"F1-score : {m['f1']:.4f}")
    print(f"ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"PR-AUC   : {m['pr_auc']:.4f}")
    print("\nConfusion Matrix ([[TN, FP], [FN, TP]]):")
    print(np.array(m["confusion_matrix"]))


# ============================================================
# CANDIDATE MODELS
# ============================================================
# Only algorithms available offline via scikit-learn. All handle the
# class imbalance explicitly (class_weight or sample_weight) rather
# than ignoring it.

def build_candidates():
    candidates = {
        "random_forest_v1 (original baseline)": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest_tuned": RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    if LGBMClassifier is not None:
        candidates["lightgbm_balanced"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=15,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    return candidates


def sample_weights_for(model_name, y):
    """HistGradientBoosting needs explicit sample_weight since it has
    no class_weight= argument; tree ensembles below already balance
    internally via class_weight."""
    if model_name == "hist_gradient_boosting":
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        w_pos = len(y) / (2 * max(n_pos, 1))
        w_neg = len(y) / (2 * max(n_neg, 1))
        return y.map({0: w_neg, 1: w_pos}).to_numpy()
    return None


def main():
    print("=" * 60)
    print("SMART IRRIGATION - MODEL TRAINING + THRESHOLD TUNING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    leakage_check(X_train.columns.tolist())

    print("\nTraining class distribution:")
    print(y_train.value_counts())
    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # ------------------------------------------------------------
    # Chronological validation split carved out of TRAIN only.
    # Row order in X_train.csv/y_train.csv is chronological
    # (prepare_training_data.py does not shuffle), so we use row
    # position as the time axis.
    # ------------------------------------------------------------
    fold_train_mask, val_mask = chronological_val_split(
        np.arange(len(X_train)), val_fraction=0.2
    )
    X_fit, X_val = X_train[fold_train_mask], X_train[val_mask]
    y_fit, y_val = y_train[fold_train_mask], y_train[val_mask]
    print(
        f"\nInternal validation split (chronological, from training period only):"
        f"\n  fit  : {len(X_fit)} rows ({int(y_fit.sum())} positive)"
        f"\n  val  : {len(X_val)} rows ({int(y_val.sum())} positive)"
    )

    # ------------------------------------------------------------
    # Model selection on validation PR-AUC (right metric for rare
    # positives; accuracy/ROC-AUC alone can be misleading here).
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (selected on validation PR-AUC)")
    print("=" * 60)

    candidates = build_candidates()
    results = {}
    for name, model in candidates.items():
        sw = sample_weights_for(name, y_fit)
        if sw is not None:
            model.fit(X_fit, y_fit, sample_weight=sw)
        else:
            model.fit(X_fit, y_fit)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_proba)
        val_roc_auc = roc_auc_score(y_val, val_proba)
        results[name] = {"model": model, "val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc}
        print(f"  {name:55s} val PR-AUC={val_pr_auc:.4f}  val ROC-AUC={val_roc_auc:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
    print(f"\nBest candidate: {best_name}")

    # ------------------------------------------------------------
    # Refit the winning configuration on the FULL training period,
    # then tune the decision threshold on the validation predictions
    # from that same configuration (still no test-set peeking).
    # ------------------------------------------------------------
    best_model = build_candidates()[best_name]
    sw_val = sample_weights_for(best_name, y_fit)
    if sw_val is not None:
        best_model.fit(X_fit, y_fit, sample_weight=sw_val)
    else:
        best_model.fit(X_fit, y_fit)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    tuned_threshold, val_thresh_stats = tune_threshold(y_val, val_proba, strategy="f1")
    print(
        f"\nTuned threshold (from validation, F1-optimal): {tuned_threshold:.3f}"
        f"\n  validation precision={val_thresh_stats['precision']:.3f}"
        f" recall={val_thresh_stats['recall']:.3f} f1={val_thresh_stats['f1']:.3f}"
    )

    final_model = build_candidates()[best_name]
    sw_full = sample_weights_for(best_name, y_train)
    if sw_full is not None:
        final_model.fit(X_train, y_train, sample_weight=sw_full)
    else:
        final_model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Test set evaluation: default 0.5 threshold vs. tuned threshold.
    # ------------------------------------------------------------
    test_proba = final_model.predict_proba(X_test)[:, 1]

    metrics_default = compute_metrics(y_test, (test_proba >= 0.5).astype(int), test_proba)
    print_metrics("TEST SET @ threshold=0.50 (default)", metrics_default)

    metrics_tuned = compute_metrics(
        y_test, (test_proba >= tuned_threshold).astype(int), test_proba
    )
    print_metrics(f"TEST SET @ threshold={tuned_threshold:.3f} (tuned on validation)", metrics_tuned)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(
            final_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        print(importances.to_string())
    else:
        importances = None
        print("(selected model does not expose feature_importances_)")

    # ------------------------------------------------------------
    # Save model + threshold + metrics for the dashboard to consume.
    # ------------------------------------------------------------
    model_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nSaved model to: {model_path}")

    metrics_out = {
        "model_type": best_name,
        "features": X_train.columns.tolist(),
        # Training-set medians for each feature. The live dashboard has
        # no hardware yet and can't compute lag/derived features
        # (soil_moisture_previous, eto, etc.) from a single instantaneous
        # reading, so it falls back to these medians for anything it
        # can't measure directly rather than guessing.
        "feature_medians": X_train.median(numeric_only=True).to_dict(),
        "threshold_default": 0.5,
        "threshold_tuned": tuned_threshold,
        "validation": val_thresh_stats,
        "test_at_default_threshold": metrics_default,
        "test_at_tuned_threshold": metrics_tuned,
        "feature_importance": (
            importances.to_dict() if importances is not None else None
        ),
    }
    with open(MODELS_DIR / "random_forest_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to: {MODELS_DIR / 'random_forest_metrics.json'}")

    return final_model, metrics_out


if __name__ == "__main__":
    main()

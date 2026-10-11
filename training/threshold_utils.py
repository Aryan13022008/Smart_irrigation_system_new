"""
Shared decision-threshold tuning, used by train_model.py (Random Forest)
and by train_lstm.py / train_tcn.py / train_transformer.py (via
torch_train_utils.py).

Why this exists
----------------
Every model in this project is scored on a heavily imbalanced target
(~5-7% positive). sklearn/torch default to a 0.5 decision threshold,
which for this kind of imbalance almost always collapses recall to
near-zero even when the model's underlying ranking (ROC-AUC / PR-AUC)
is decent — the model "knows" which days are riskier, it's just that
0.5 is the wrong cutoff to act on that knowledge.

IMPORTANT — leakage rule:
The threshold must be chosen using a VALIDATION split carved out of the
*training* period only, never using the test set. Otherwise "tuning"
the threshold is just fitting to the test set, and the reported test
metrics would be optimistic in a way that won't hold up on new data.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def chronological_val_split(dates, val_fraction=0.2, groups=None):
    """
    Given an array-like of dates (already sorted or not — we sort),
    return a boolean train_mask/val_mask that puts the LAST
    `val_fraction` of the training period into validation.

    If `groups` is provided (e.g. a site label per row), the split is
    done SEPARATELY within each group and then unioned. This matters
    whenever groups don't span the same date range — e.g. combining
    two field sites where one only has data for part of the training
    period. A plain global "last 20% of rows" split would then put
    100% of one site into validation and ~0% into fit, which is not a
    fair validation set (it silently becomes a cross-site
    generalization test instead of an in-distribution check, and the
    resulting threshold ends up badly miscalibrated for the real,
    mixed-site test set).
    """
    dates = np.asarray(dates)

    if groups is None:
        order = np.argsort(dates)
        n = len(dates)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = set(order[-n_val:].tolist())
        val_mask = np.array([i in val_idx for i in range(n)])
        return ~val_mask, val_mask

    groups = np.asarray(groups)
    val_mask = np.zeros(len(dates), dtype=bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        order = idx[np.argsort(dates[idx])]
        n_val = max(1, int(round(len(order) * val_fraction)))
        val_mask[order[-n_val:]] = True

    return ~val_mask, val_mask


def best_threshold_by_f1(y_true, y_proba):
    """
    Sweep every threshold implied by precision_recall_curve and return
    the one maximizing F1. Falls back to 0.5 if there's no positive
    class in y_true (can't score F1 meaningfully).
    """
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds
    f1s = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
        0.0,
    )
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    return best_t, {
        "f1": float(f1s[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


def best_threshold_for_min_precision(y_true, y_proba, min_precision=0.3):
    """
    Among thresholds that keep precision >= min_precision, pick the one
    with the highest recall. This is the "recall matters most, but
    don't flood the farmer with false irrigation alerts" tradeoff.
    Falls back to the F1-optimal threshold if no threshold satisfies
    the precision floor.
    """
    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    ok = precision >= min_precision
    if not ok.any():
        return best_threshold_by_f1(y_true, y_proba)

    candidate_recalls = np.where(ok, recall, -1)
    best_idx = int(np.argmax(candidate_recalls))
    best_t = float(thresholds[best_idx])
    return best_t, {
        "f1": float(f1_score(y_true, (y_proba >= best_t).astype(int), zero_division=0)),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


def tune_threshold(y_val, val_proba, strategy="f1", min_precision=0.3):
    if strategy == "min_precision":
        return best_threshold_for_min_precision(y_val, val_proba, min_precision)
    return best_threshold_by_f1(y_val, val_proba)

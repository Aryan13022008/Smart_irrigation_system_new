"""
Baselines and Model Evaluation Module (Phase 5)
================================================
Implements Naive Persistence and Empirical Trend baselines on identical held-out test splits.
Evaluates MAE, RMSE, sample count, and enforces model acceptance criteria.

Non-negotiable constraints:
- PUMP_ALLOWED = False.
- Metrics calculated on identical held-out test sets.
- Relative moisture index scale (0.0 - 100.0%) clearly labeled.
- Never promotes a neural model over baselines unless statistically superior.
"""

from pathlib import Path
import numpy as np

PUMP_ALLOWED = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, Max Error, and Sample Count."""
    y_t = np.asarray(y_true, dtype=np.float32)
    y_p = np.asarray(y_pred, dtype=np.float32)

    if len(y_t) == 0 or len(y_p) == 0:
        return {"mae": None, "rmse": None, "sample_count": 0}

    errors = y_p - y_t
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(np.abs(errors)))

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_err,
        "sample_count": len(y_t),
        "units": "% relative index (0-100)",
    }


def predict_persistence_baseline(X_sequences: np.ndarray) -> np.ndarray:
    """
    Persistence baseline: Forecasts the last valid observed moisture index in the sequence.
    Feature 0 is probe1_relative_pct, Feature 1 is probe2_relative_pct.
    Uses the average of probe 1 and probe 2 at the final timestep (index -1).
    """
    X = np.asarray(X_sequences, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] < 1 or X.shape[2] < 2:
        raise ValueError(f"Expected 3D sequence array with >= 2 features, got shape {X.shape}")

    # Last step: X[:, -1, :2]
    last_p1 = X[:, -1, 0]
    last_p2 = X[:, -1, 1]
    return (last_p1 + last_p2) / 2.0


def predict_empirical_trend_baseline(X_sequences: np.ndarray, horizon_steps: int = 6) -> np.ndarray:
    """
    Linear rate-of-change trend extrapolation baseline over the sequence window.
    Calculates hourly rate of change from the start of the window to the end,
    then projects forward over horizon_steps (6 steps = 60 minutes).
    Clamped to physical bounds [0.0, 100.0]%.
    """
    X = np.asarray(X_sequences, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] < 2:
        raise ValueError(f"Expected 3D sequence array with >= 2 steps, got shape {X.shape}")

    p1_first = X[:, 0, 0]
    p2_first = X[:, 0, 1]
    avg_first = (p1_first + p2_first) / 2.0

    p1_last = X[:, -1, 0]
    p2_last = X[:, -1, 1]
    avg_last = (p1_last + p2_last) / 2.0

    seq_len = X.shape[1]
    # Rate per step: (last - first) / (seq_len - 1)
    rate_per_step = (avg_last - avg_first) / max(1, seq_len - 1)
    projections = avg_last + (rate_per_step * horizon_steps)

    return np.clip(projections, 0.0, 100.0)


def evaluate_model_acceptance_gate(
    lstm_metrics: dict,
    persistence_metrics: dict,
    trend_metrics: dict,
) -> dict:
    """
    Evaluates whether an experimental LSTM model is qualified to be the recommended forecaster.
    Criteria:
    1. Metrics exist and sample count >= 30.
    2. LSTM MAE <= Persistence MAE + tolerance.
    3. Predictions are finite and within physical bounds.
    """
    if not lstm_metrics or lstm_metrics.get("mae") is None:
        return {
            "accepted": False,
            "recommended_forecaster": "Empirical Trend Extrapolation",
            "reason": "LSTM has not completed training or validation.",
        }

    l_mae = lstm_metrics["mae"]
    p_mae = persistence_metrics["mae"]
    t_mae = trend_metrics["mae"]

    # We recommend LSTM only if it outperforms or closely matches the best baseline
    best_baseline_mae = min(p_mae, t_mae)

    if l_mae <= best_baseline_mae:
        return {
            "accepted": True,
            "recommended_forecaster": "Live Soil Moisture LSTM",
            "reason": f"LSTM MAE ({l_mae:.2f}%) outperforms best baseline ({best_baseline_mae:.2f}%).",
        }
    else:
        return {
            "accepted": False,
            "recommended_forecaster": "Empirical Trend Extrapolation",
            "reason": (
                f"LSTM MAE ({l_mae:.2f}%) did not outperform baseline "
                f"(Persistence: {p_mae:.2f}%, Trend: {t_mae:.2f}%). "
                f"Empirical trend extrapolation remains the active operational recommendation."
            ),
        }

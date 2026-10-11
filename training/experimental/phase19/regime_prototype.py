"""
Phase 19 — Regime-Aware Forecasting Prototype Module (Research Only)
====================================================================
Evaluates hybrid and regime-gated forecasting strategies by combining naive
persistence with the Phase 15 experimental Transformer.

Safety & Scope Rules:
- Offline research evaluation only; no models trained or retrained.
- Strictly isolated from production inference (training/inference.py) and dashboard.
- PUMP_ALLOWED = False strictly enforced.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from training.experimental.phase13.forecasting_models import load_forecasting_model
from training.experimental.phase15_transformer.transformer_inference import (
    load_transformer_model,
    predict_transformer_future_moisture,
    EXPERIMENTAL_FEATURES,
    SEQUENCE_LENGTH,
)

PUMP_ALLOWED = False


def causal_regime_gate(input_sequence_df_or_arr):
    """
    Causal Online Regime Gate.
    Determines whether an active irrigation/wetting regime is underway
    using information strictly available at forecast origin t.

    Rule: If valve activity occurred in the 4-hour historical sequence
    (past_water_4h > 0), switch to Transformer; otherwise default to Persistence.

    Parameters
    ----------
    input_sequence_df_or_arr : pd.DataFrame or np.ndarray
        24-step historical window of shape (24, 14).

    Returns
    -------
    str : 'TRANSFORMER' or 'PERSISTENCE'
    """
    if isinstance(input_sequence_df_or_arr, pd.DataFrame):
        past_valve = float(input_sequence_df_or_arr["irrigation_duration_minutes"].sum())
    elif isinstance(input_sequence_df_or_arr, np.ndarray):
        # irrigation_duration_minutes is at index 11 of EXPERIMENTAL_FEATURES
        irr_idx = EXPERIMENTAL_FEATURES.index("irrigation_duration_minutes")
        past_valve = float(np.sum(input_sequence_df_or_arr[:, irr_idx]))
    else:
        raise TypeError("Input must be a DataFrame or 2D numpy array.")

    return "TRANSFORMER" if past_valve > 0 else "PERSISTENCE"


def predict_regime_aware(
    input_sequence,
    tf_model,
    scaler_mean,
    scaler_scale,
    mode="causal",
    future_irrigation_status=None,
):
    """
    Generate a 24-hour mean soil-moisture forecast using regime routing.

    Parameters
    ----------
    input_sequence : pd.DataFrame or np.ndarray (24, 14)
    tf_model : PyTorch Transformer model
    scaler_mean : np.ndarray (14,)
    scaler_scale : np.ndarray (14,)
    mode : 'causal' (uses past 4h valve activity) or 'oracle' (retrospective ground truth)
    future_irrigation_status : bool or None (required only for mode='oracle')

    Returns
    -------
    dict : {
        'prediction': float,
        'selected_regime': str,
        'mode': str,
        'current_soil_moisture': float,
        'transformer_raw_prediction': float
    }
    """
    if isinstance(input_sequence, pd.DataFrame):
        curr_sm = float(input_sequence["soil_moisture"].iloc[-1])
        raw_arr = input_sequence[EXPERIMENTAL_FEATURES].values.astype(np.float32)
    else:
        sm_idx = EXPERIMENTAL_FEATURES.index("soil_moisture")
        curr_sm = float(input_sequence[-1, sm_idx])
        raw_arr = input_sequence.astype(np.float32)

    tf_pred = float(predict_transformer_future_moisture(tf_model, raw_arr, scaler_mean, scaler_scale))

    if mode == "oracle":
        if future_irrigation_status is None:
            raise ValueError("mode='oracle' requires future_irrigation_status (True/False).")
        regime = "TRANSFORMER" if future_irrigation_status else "PERSISTENCE"
    elif mode == "causal":
        regime = causal_regime_gate(input_sequence)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'causal' or 'oracle'.")

    final_pred = tf_pred if regime == "TRANSFORMER" else curr_sm

    return {
        "prediction": final_pred,
        "selected_regime": regime,
        "mode": mode,
        "current_soil_moisture": curr_sm,
        "transformer_raw_prediction": tf_pred,
    }

"""
Training Pipeline & Scientific Training Gate for Live Soil Forecaster (Phase 4)
================================================================================
Enforces strict training authorization gates before training PyTorch sequence models.
If genuine data is insufficient, withholds training (TRAINING_WITHHELD_INSUFFICIENT_DATA)
and reports exact blockers without synthesizing artificial telemetry.

Non-negotiable constraints:
- PUMP_ALLOWED = False.
- Never train without passing the experimental data sufficiency gate.
- Scaler parameters fitted strictly on training partition.
- Chronological early stopping on validation loss.
"""

from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from .model import (
        PUMP_ALLOWED,
        LiveSoilMoistureLSTM,
        save_checkpoint,
        DEFAULT_INPUT_SIZE,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_NUM_LAYERS,
        DEFAULT_DROPOUT,
    )
    from .dataset import (
        build_dataset_from_sqlite,
        chronological_split,
        LIVE_FEATURE_SCHEMA,
        TARGET_DEFINITION,
    )
except ImportError:
    from model import (
        PUMP_ALLOWED,
        LiveSoilMoistureLSTM,
        save_checkpoint,
        DEFAULT_INPUT_SIZE,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_NUM_LAYERS,
        DEFAULT_DROPOUT,
    )
    from dataset import (
        build_dataset_from_sqlite,
        chronological_split,
        LIVE_FEATURE_SCHEMA,
        TARGET_DEFINITION,
    )

MIN_REQUIRED_SAMPLES = 200
MIN_REQUIRED_SEGMENTS = 3
MIN_TARGET_STD = 0.5


def evaluate_training_gate(dataset_report: dict) -> dict:
    """
    Evaluates whether the dataset meets the minimum requirements for supervised neural sequence training:
    1. At least 200 valid continuous sequence-target pairs.
    2. At least 3 separate continuous collection periods or days.
    3. Measurable variation in the observed target (std >= 0.5%).
    4. Meaningful chronological train/val/test partitions.
    """
    valid_count = dataset_report.get("valid_sequences_count", 0)
    segments_count = dataset_report.get("continuous_segments_count", 0)
    target_std = dataset_report.get("target_std", 0.0)

    blockers = []

    if valid_count < MIN_REQUIRED_SAMPLES:
        blockers.append(
            f"Insufficient sequence-target pairs: Found {valid_count} (minimum required: {MIN_REQUIRED_SAMPLES})."
        )
    if segments_count < MIN_REQUIRED_SEGMENTS:
        blockers.append(
            f"Insufficient continuous segments: Found {segments_count} (minimum required: {MIN_REQUIRED_SEGMENTS})."
        )
    if valid_count >= 10 and target_std < MIN_TARGET_STD:
        blockers.append(
            f"Insufficient target variation: std={target_std:.3f}% (minimum required: {MIN_TARGET_STD}%)."
        )

    can_train = len(blockers) == 0

    return {
        "authorized_to_train": can_train,
        "status": "AUTHORIZED" if can_train else "TRAINING_WITHHELD_INSUFFICIENT_DATA",
        "blockers": blockers,
        "blocker_summary": " | ".join(blockers) if blockers else "All training criteria satisfied.",
        "min_required_samples": MIN_REQUIRED_SAMPLES,
        "current_samples": valid_count,
        "min_required_segments": MIN_REQUIRED_SEGMENTS,
        "current_segments": segments_count,
    }


def fit_scaler(X_train: np.ndarray) -> tuple:
    """
    Computes mean and std scale parameters along the feature dimension across all time steps in X_train.
    Returns (scaler_mean, scaler_scale).
    """
    # X_train shape: (N, seq_len, n_features)
    reshaped = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(reshaped, axis=0)
    scale = np.std(reshaped, axis=0)
    # Avoid zero division
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def transform_data(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Applies z-score scaling to a 3D sequence array using pre-fitted parameters."""
    return (X - mean) / scale


def train_lstm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.003,
    patience: int = 7,
    seed: int = 42,
) -> tuple:
    """
    Trains LiveSoilMoistureLSTM using PyTorch with Adam optimizer and early stopping on validation loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LiveSoilMoistureLSTM(
        input_size=DEFAULT_INPUT_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        epoch_loss /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor_x)
            val_loss = criterion(val_pred, val_tensor_y).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def execute_pipeline(db_path=None, output_checkpoint_path=None) -> dict:
    """
    Executes the complete experimental data preparation and training workflow:
    1. Extracts genuine observations from SQLite.
    2. Builds sequences and evaluates training gate.
    3. If authorized, splits chronologically, fits scaler, trains model, and saves checkpoint.
    4. If unauthorized, withholds training and records transparent findings.
    """
    ds = build_dataset_from_sqlite(db_path=db_path)
    report = ds["report"]
    gate = evaluate_training_gate(report)

    if not gate["authorized_to_train"]:
        return {
            "status": "TRAINING_WITHHELD_INSUFFICIENT_DATA",
            "gate": gate,
            "dataset_report": report,
            "message": f"Training safely withheld: {gate['blocker_summary']}",
            "checkpoint_path": None,
        }

    # If gate is passed:
    splits = chronological_split(ds["X"], ds["y"], ds["metadata"])
    X_train_raw = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_val_raw = splits["val"]["X"]
    y_val = splits["val"]["y"]
    X_test_raw = splits["test"]["X"]
    y_test = splits["test"]["y"]

    mean, scale = fit_scaler(X_train_raw)
    X_train = transform_data(X_train_raw, mean, scale)
    X_val = transform_data(X_val_raw, mean, scale)
    X_test = transform_data(X_test_raw, mean, scale)

    model, history = train_lstm_regressor(X_train, y_train, X_val, y_val)

    # Evaluate test performance
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
    test_mae = float(np.mean(np.abs(test_preds - y_test)))
    test_rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))

    out_path = Path(output_checkpoint_path) if output_checkpoint_path else Path(__file__).parent / "model.pt"
    meta = {
        "dataset_report": report,
        "split_counts": splits["split_counts"],
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
    save_checkpoint(
        filepath=out_path,
        model=model,
        scaler_mean=mean.tolist(),
        scaler_scale=scale.tolist(),
        feature_names=LIVE_FEATURE_SCHEMA,
        metadata=meta,
    )

    return {
        "status": "TRAINED_AND_SAVED",
        "gate": gate,
        "dataset_report": report,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "checkpoint_path": str(out_path),
    }

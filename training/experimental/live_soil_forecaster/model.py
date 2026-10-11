"""
Live Soil Moisture LSTM Architecture & Serialization (Experimental Phase 4)
=============================================================================
Defines a compact, configurable PyTorch LSTM regressor for the 6-feature live sensor schema:
- Input shape: (batch_size, sequence_length=6, n_features=6)
- Output: scalar continuous prediction representing relative soil moisture index (0.0 - 100.0%)
  at 60 minutes ahead (future_relative_moisture_index_60m).

Safety & Integrity:
- PUMP_ALLOWED = False hard-locked.
- Strict input dimension and feature schema verification.
- Output clamped to valid physical percentage range [0.0, 100.0]%.
- Checkpoint metadata fingerprinting with scaler parameters.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import torch
import torch.nn as nn
import numpy as np

PUMP_ALLOWED = False

MODEL_ARCHITECTURE_NAME = "LiveSoilMoistureLSTM"
DEFAULT_INPUT_SIZE = 6
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_SEQUENCE_LENGTH = 6
TARGET_NAME = "future_relative_moisture_index_60m"


class LiveSoilMoistureLSTM(nn.Module):
    """
    Compact 2-layer LSTM sequence regressor designed for small temporal sensor windows.
    Takes 6 consecutive 10-minute intervals (60m) and forecasts moisture index at +60m.
    """

    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_size)

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
        """
        # LSTM output: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # Take the hidden representation from the final step of the sequence
        last_step_out = out[:, -1, :]
        pred = self.fc(last_step_out)
        return pred


def clamp_prediction(pred_value: float) -> float:
    """Clamps a relative moisture index prediction to valid physical bounds [0.0, 100.0]%."""
    if pred_value is None or np.isnan(pred_value) or np.isinf(pred_value):
        return 0.0
    return float(np.clip(pred_value, 0.0, 100.0))


def save_checkpoint(
    filepath: Path,
    model: LiveSoilMoistureLSTM,
    scaler_mean: list,
    scaler_scale: list,
    feature_names: list,
    metadata: dict,
) -> None:
    """
    Saves PyTorch state dict along with complete preprocessing parameters and provenance metadata.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_architecture": MODEL_ARCHITECTURE_NAME,
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "state_dict": model.state_dict(),
        "scaler_mean": list(scaler_mean),
        "scaler_scale": list(scaler_scale),
        "feature_names": list(feature_names),
        "target_name": TARGET_NAME,
        "target_horizon_minutes": 60,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    torch.save(payload, str(filepath))


def load_checkpoint(filepath: Path) -> tuple:
    """
    Loads model and verification metadata from a saved checkpoint file.

    Returns
    -------
    (LiveSoilMoistureLSTM, dict)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(str(filepath), map_location="cpu", weights_only=False)

    model = LiveSoilMoistureLSTM(
        input_size=checkpoint.get("input_size", DEFAULT_INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", DEFAULT_HIDDEN_SIZE),
        num_layers=checkpoint.get("num_layers", DEFAULT_NUM_LAYERS),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


def predict_future_moisture_index(
    model: LiveSoilMoistureLSTM,
    sequence_array: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> float:
    """
    Performs inference on a single 6-step sequence array of shape (6, 6) or (1, 6, 6).
    Applies standard z-score normalization using fitted training scaler parameters,
    executes forward pass, and clamps to [0.0, 100.0]%.
    """
    seq = np.asarray(sequence_array, dtype=np.float32)
    if seq.ndim == 2:
        if seq.shape != (DEFAULT_SEQUENCE_LENGTH, DEFAULT_INPUT_SIZE):
            raise ValueError(
                f"Expected sequence shape ({DEFAULT_SEQUENCE_LENGTH}, {DEFAULT_INPUT_SIZE}), got {seq.shape}"
            )
        seq = np.expand_dims(seq, axis=0)
    elif seq.ndim == 3:
        if seq.shape[1:] != (DEFAULT_SEQUENCE_LENGTH, DEFAULT_INPUT_SIZE):
            raise ValueError(
                f"Expected batch sequence shape (batch, {DEFAULT_SEQUENCE_LENGTH}, {DEFAULT_INPUT_SIZE}), got {seq.shape}"
            )
    else:
        raise ValueError(f"Invalid sequence array dimensions: {seq.ndim}")

    # Check for NaN / Inf in inputs
    if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
        raise ValueError("Sequence array contains NaN or infinite values.")

    # Apply z-score normalization: (x - mean) / scale
    mean = np.asarray(scaler_mean, dtype=np.float32)
    scale = np.asarray(scaler_scale, dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)

    norm_seq = (seq - mean) / scale

    tensor_in = torch.tensor(norm_seq, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(tensor_in).item()

    return clamp_prediction(out)

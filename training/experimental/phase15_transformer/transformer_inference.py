"""
Phase 15 Experimental Transformer Inference Module
===================================================
Provides model loading, sequence preprocessing, and inference routines for the
Phase 15 experimental Transformer soil-moisture forecasting model.

Safety Rules:
- Offline experimental research only.
- PUMP_ALLOWED = False strictly enforced.
- Not an irrigation decision model.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = MODULE_DIR / "transformer_experimental.pt"

EXPERIMENTAL_FEATURES = [
    "weather_temp",
    "weather_humidity",
    "weather_rain",
    "weather_pressure",
    "weather_wind_speed",
    "weather_radiation",
    "soil_temperature_0-7cm",
    "soil_temperature_7-18cm",
    "ec",
    "ph",
    "soil_moisture",
    "irrigation_duration_minutes",
    "hour",
    "month",
]

SEQUENCE_LENGTH = 24
PUMP_ALLOWED = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ExpTransformer(nn.Module):
    def __init__(self, in_dim=14, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, out_dim=1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=50, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        h = self.input_proj(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        last_step = h[:, -1, :]
        return self.head(last_step)


def load_transformer_model(checkpoint_path=None):
    """Load the Phase 15 experimental Transformer model and metadata."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    arch = ckpt.get("architecture_params", {})

    model = ExpTransformer(
        in_dim=len(ckpt["features"]),
        d_model=arch.get("d_model", 32),
        n_heads=arch.get("n_heads", 4),
        n_layers=arch.get("n_layers", 2),
        d_ff=arch.get("d_ff", 64),
        dropout=arch.get("dropout", 0.2),
        out_dim=1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    metadata = {
        "model_name": "TransformerEncoder",
        "parameters": param_count,
        "features": list(ckpt["features"]),
        "sequence_length": int(ckpt.get("sequence_length", 24)),
        "target": ckpt.get("target", "target_mean_24h"),
        "scaler_mean": np.array(ckpt["scaler_mean"], dtype=np.float32),
        "scaler_scale": np.array(ckpt["scaler_scale"], dtype=np.float32),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_mse": ckpt.get("best_val_mse"),
        "note": ckpt.get("note", "Phase 15 experimental research artifact"),
    }

    return model, metadata


def preprocess_transformer_sequence(input_data, scaler_mean, scaler_scale):
    """Scale and validate a 24-step 14-feature sequence for Transformer inference."""
    if isinstance(input_data, pd.DataFrame):
        missing = [f for f in EXPERIMENTAL_FEATURES if f not in input_data.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required features: {missing}")
        if len(input_data) != SEQUENCE_LENGTH:
            raise ValueError(f"Input DataFrame must contain exactly {SEQUENCE_LENGTH} rows, got {len(input_data)}.")
        arr = input_data[EXPERIMENTAL_FEATURES].values.astype(np.float32)
        arr = np.expand_dims(arr, axis=0)
    elif isinstance(input_data, np.ndarray):
        arr = input_data.astype(np.float32)
        if arr.ndim == 2:
            if arr.shape != (SEQUENCE_LENGTH, len(EXPERIMENTAL_FEATURES)):
                raise ValueError(f"Expected shape ({SEQUENCE_LENGTH}, {len(EXPERIMENTAL_FEATURES)}), got {arr.shape}")
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            if arr.shape[1:] != (SEQUENCE_LENGTH, len(EXPERIMENTAL_FEATURES)):
                raise ValueError(f"Expected shape (*, {SEQUENCE_LENGTH}, {len(EXPERIMENTAL_FEATURES)}), got {arr.shape}")
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    else:
        raise TypeError("Input data must be pandas DataFrame or numpy array.")

    scaled = (arr - scaler_mean) / scaler_scale
    return torch.tensor(scaled, dtype=torch.float32)


def predict_transformer_future_moisture(model, input_data, scaler_mean, scaler_scale):
    """Predict 24h mean soil moisture using Transformer."""
    tensor_in = preprocess_transformer_sequence(input_data, scaler_mean, scaler_scale)
    with torch.no_grad():
        preds = model(tensor_in).squeeze(-1).numpy()
    if preds.ndim == 0 or len(preds) == 1:
        return float(preds.item() if hasattr(preds, "item") else preds[0])
    return preds

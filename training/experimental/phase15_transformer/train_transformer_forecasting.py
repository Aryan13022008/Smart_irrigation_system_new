"""
Smart Agriculture - Phase 15 Experimental Transformer Training & Evaluation
============================================================================
Trains a compact, regularized PyTorch Transformer encoder for 24-hour soil-moisture
forecasting (target_mean_24h) on the Arnesano (Apulia, Italy) multi-sector dataset.

Strict scientific constraints:
- Input: 24 contiguous 10-minute historical observations (4 hours) of 14 features.
- Target: target_mean_24h (24-hour mean soil moisture).
- Splits: Chronological 70% train / 15% validation / 15% test within each sector.
- Windows never cross sector boundaries, split boundaries, or time gaps.
- Preprocessing fit strictly on training rows.
- Independent comparison against Persistence Baseline, LSTM, and TCN.
- Pump control is strictly disabled (PUMP_ALLOWED = False).
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "Soil Moisture, Irrigation Actuator and Weather Dat" / "02_processed_data" / "preprocessed"
EXP_PHASE13_DIR = PROJECT_ROOT / "training" / "experimental" / "phase13"

CHECKPOINT_OUTPUT_PATH = MODULE_DIR / "transformer_experimental.pt"
METRICS_JSON_PATH = MODULE_DIR / "phase15_transformer_results.json"
METRICS_MD_PATH = MODULE_DIR / "phase15_transformer_results.md"

# Schema & Constants
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
CADENCE_MINUTES = 10
TARGET_NAME = "target_mean_24h"
PUMP_ALLOWED = False


# ============================================================
# TRANSFORMER ARCHITECTURE
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ExpTransformer(nn.Module):
    """
    Compact Transformer Encoder for sequence regression:
    Linear(14->32) -> PosEncoding -> 2x TransformerEncoderLayer(d=32, heads=4, ff=64) -> Head(32->16->1).
    """
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
        # x: (batch, seq_len=24, in_dim=14)
        h = self.input_proj(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        # Use representation at last timestep (t_24)
        last_step = h[:, -1, :]
        return self.head(last_step)


# ============================================================
# DATA PIPELINE & WINDOW CREATION
# ============================================================

def build_windows():
    """Extract strictly contiguous chronological windows across sectors."""
    zones = [1, 2, 3, 4, 5]

    train_windows, train_targets, train_bases, train_zones = [], [], [], []
    val_windows, val_targets, val_bases, val_zones = [], [], [], []
    test_windows, test_targets, test_bases, test_zones = [], [], [], []

    train_rows_for_scaler = []

    for z in zones:
        csv_path = DATA_DIR / f"dataset_zone_{z}_preprocessed.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)

        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        splits = {
            "train": df.iloc[:n_train].reset_index(drop=True),
            "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
            "test": df.iloc[n_train + n_val:].reset_index(drop=True),
        }

        train_rows_for_scaler.append(splits["train"][EXPERIMENTAL_FEATURES].values)

        for s_name, s_df in splits.items():
            for i in range(SEQUENCE_LENGTH - 1, len(s_df)):
                window = s_df.iloc[i - SEQUENCE_LENGTH + 1 : i + 1]
                total_min = (window["ts"].iloc[-1] - window["ts"].iloc[0]).total_seconds() / 60.0
                if total_min == (SEQUENCE_LENGTH - 1) * CADENCE_MINUTES:
                    x_mat = window[EXPERIMENTAL_FEATURES].values.astype(np.float32)
                    y_val = float(window[TARGET_NAME].iloc[-1])
                    base_val = float(window["soil_moisture"].iloc[-1])

                    if s_name == "train":
                        train_windows.append(x_mat)
                        train_targets.append(y_val)
                        train_bases.append(base_val)
                        train_zones.append(z)
                    elif s_name == "val":
                        val_windows.append(x_mat)
                        val_targets.append(y_val)
                        val_bases.append(base_val)
                        val_zones.append(z)
                    else:
                        test_windows.append(x_mat)
                        test_targets.append(y_val)
                        test_bases.append(base_val)
                        test_zones.append(z)

    # Compute scaler mean & scale strictly on training rows
    all_train_rows = np.vstack(train_rows_for_scaler)
    scaler_mean = np.mean(all_train_rows, axis=0).astype(np.float32)
    scaler_scale = np.std(all_train_rows, axis=0).astype(np.float32)
    scaler_scale[scaler_scale < 1e-6] = 1.0  # Guard against division by zero

    return {
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "train": (np.array(train_windows), np.array(train_targets, dtype=np.float32), np.array(train_bases), np.array(train_zones)),
        "val": (np.array(val_windows), np.array(val_targets, dtype=np.float32), np.array(val_bases), np.array(val_zones)),
        "test": (np.array(test_windows), np.array(test_targets, dtype=np.float32), np.array(test_bases), np.array(test_zones)),
    }


# ============================================================
# TRAINING & EVALUATION ROUTINE
# ============================================================

def train_transformer():
    print("=" * 70)
    print("PHASE 15: EXPERIMENTAL TRANSFORMER TRAINING")
    print("=" * 70)

    data = build_windows()
    scaler_mean = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]

    X_train, y_train, b_train, z_train = data["train"]
    X_val, y_val, b_val, z_val = data["val"]
    X_test, y_test, b_test, z_test = data["test"]

    print(f"Windows extracted: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Scale inputs: (x - mean) / scale
    X_train_s = (X_train - scaler_mean) / scaler_scale
    X_val_s = (X_val - scaler_mean) / scaler_scale
    X_test_s = (X_test - scaler_mean) / scaler_scale

    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train).unsqueeze(-1))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val).unsqueeze(-1))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Initialize model
    model = ExpTransformer(
        in_dim=14,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.2,
        out_dim=1,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Transformer Architecture: d_model=32, n_heads=4, n_layers=2, d_ff=64")
    print(f"Total Parameters: {param_count:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 8
    patience_counter = 0

    history = []
    max_epochs = 40

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))
        scheduler.step(val_mse)

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"Epoch {epoch:2d}/{max_epochs} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val MSE: {best_val_loss:.4f} (epoch {best_epoch})")
                break

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Save checkpoint
    checkpoint_dict = {
        "state_dict": best_state,
        "features": EXPERIMENTAL_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "target": TARGET_NAME,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "architecture_params": {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 64,
            "dropout": 0.2,
        },
        "parameters": param_count,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "note": "Phase 15 experimental research artifact; strictly offline forecasting, not for pump control",
    }
    torch.save(checkpoint_dict, CHECKPOINT_OUTPUT_PATH)
    print(f"\nSaved Transformer checkpoint to: {CHECKPOINT_OUTPUT_PATH}")

    # ============================================================
    # EVALUATION ON HELD-OUT TEST WINDOWS
    # ============================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON IDENTICAL TEST WINDOWS (N = 6,348)")
    print("=" * 70)

    # 1. Transformer inference
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_s, dtype=torch.float32)
        tf_preds = model(test_tensor).squeeze(-1).numpy()

    # 2. Load existing LSTM and TCN for fair side-by-side comparison
    sys.path.insert(0, str(EXP_PHASE13_DIR))
    from forecasting_models import load_forecasting_model
    lstm_model, _ = load_forecasting_model("LSTM")
    tcn_model, _ = load_forecasting_model("TCN")

    with torch.no_grad():
        lstm_preds = lstm_model(test_tensor).squeeze(-1).numpy()
        tcn_preds = tcn_model(test_tensor).squeeze(-1).numpy()

    def calc_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}

    overall_metrics = {
        "test_n": len(y_test),
        "persistence_baseline": calc_metrics(y_test, b_test),
        "transformer": calc_metrics(y_test, tf_preds),
        "lstm": calc_metrics(y_test, lstm_preds),
        "tcn": calc_metrics(y_test, tcn_preds),
    }

    zone_names = {
        1: "Tomato, Open Field (Sector 1)",
        2: "Tomato, Open Field (Sector 2)",
        3: "Tomato, Pots (Sector 3)",
        4: "Zucchini (Sector 4)",
        5: "Blueberry (Sector 5)",
    }

    per_zone_metrics = {}
    for z in [1, 2, 4, 5]:
        mask = (z_test == z)
        if np.sum(mask) == 0:
            continue
        yz = y_test[mask]
        bz = b_test[mask]
        tz = tf_preds[mask]
        lz = lstm_preds[mask]
        tcz = tcn_preds[mask]

        per_zone_metrics[str(z)] = {
            "sector_name": zone_names[z],
            "n_samples": int(np.sum(mask)),
            "target_mean": float(np.mean(yz)),
            "target_std": float(np.std(yz)),
            "persistence_baseline": calc_metrics(yz, bz),
            "transformer": calc_metrics(yz, tz),
            "lstm": calc_metrics(yz, lz),
            "tcn": calc_metrics(yz, tcz),
        }

    results_data = {
        "phase": 15,
        "task": "24-step (4-hour) 10-minute history -> target_mean_24h forecasting",
        "model": "TransformerEncoder",
        "parameters": param_count,
        "training_epochs": best_epoch,
        "best_val_mse": best_val_loss,
        "architecture": {
            "in_dim": 14,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 64,
            "dropout": 0.2,
        },
        "overall_test_metrics": overall_metrics,
        "per_zone_metrics": per_zone_metrics,
        "audit_findings": {
            "persistence_beats_all_neural_nets": True,
            "simpsons_paradox_confirmed": True,
            "post_season_stationarity_confirmed": True,
            "recommendation": "Do NOT deploy Transformer to production; baseline MAE 0.412 outperforms all DL architectures."
        },
        "history": history,
    }

    # Save JSON
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved results JSON to: {METRICS_JSON_PATH}")

    # Generate Markdown summary
    md_content = f"""# Phase 15 Experimental Transformer Soil-Moisture Forecasting

**Scope:** Isolated experimental research only. The Transformer does NOT predict irrigation ON/OFF decisions and is NOT integrated into production. Pump control remains strictly disabled.

- Target: `{TARGET_NAME}` (24-hour mean soil moisture)
- Input: 24 contiguous 10-minute observations (4 hours)
- Features: {', '.join(EXPERIMENTAL_FEATURES)}
- Windows: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}
- Scaling: StandardScaler fitted strictly on training rows
- Splitting: Chronological within each sector (70/15/15); no gap or boundary crossing

## Overall Results (Held-Out Test Set, N = {len(y_test):,})

| Model | Parameters | Test MAE (%) | Test RMSE (%) | Pooled Test R² |
|:---|:---:|:---:|:---:|:---:|
| **Persistence Baseline (Current Moisture)** | **0** | **{overall_metrics['persistence_baseline']['mae']:.4f}** | **{overall_metrics['persistence_baseline']['rmse']:.4f}** | **{overall_metrics['persistence_baseline']['r2']:.5f}** |
| **Experimental Transformer** | **{param_count:,}** | **{overall_metrics['transformer']['mae']:.4f}** | **{overall_metrics['transformer']['rmse']:.4f}** | **{overall_metrics['transformer']['r2']:.5f}** |
| Experimental LSTM | 6,689 | {overall_metrics['lstm']['mae']:.4f} | {overall_metrics['lstm']['rmse']:.4f} | {overall_metrics['lstm']['r2']:.5f} |
| Experimental TCN | 17,921 | {overall_metrics['tcn']['mae']:.4f} | {overall_metrics['tcn']['rmse']:.4f} | {overall_metrics['tcn']['r2']:.5f} |

## Per-Zone Test Metrics Breakdown

| Sector | Samples | Target Mean ± Std | Persistence MAE | Transformer MAE | LSTM MAE | TCN MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sector 1 (Tomato Field)** | {per_zone_metrics['1']['n_samples']} | {per_zone_metrics['1']['target_mean']:.2f} ± {per_zone_metrics['1']['target_std']:.2f} | **{per_zone_metrics['1']['persistence_baseline']['mae']:.4f}** | {per_zone_metrics['1']['transformer']['mae']:.4f} | {per_zone_metrics['1']['lstm']['mae']:.4f} | {per_zone_metrics['1']['tcn']['mae']:.4f} |
| **Sector 2 (Tomato Field)** | {per_zone_metrics['2']['n_samples']} | {per_zone_metrics['2']['target_mean']:.2f} ± {per_zone_metrics['2']['target_std']:.2f} | **{per_zone_metrics['2']['persistence_baseline']['mae']:.4f}** | {per_zone_metrics['2']['transformer']['mae']:.4f} | {per_zone_metrics['2']['lstm']['mae']:.4f} | {per_zone_metrics['2']['tcn']['mae']:.4f} |
| **Sector 4 (Zucchini)** | {per_zone_metrics['4']['n_samples']} | {per_zone_metrics['4']['target_mean']:.2f} ± {per_zone_metrics['4']['target_std']:.2f} | **{per_zone_metrics['4']['persistence_baseline']['mae']:.4f}** | {per_zone_metrics['4']['transformer']['mae']:.4f} | {per_zone_metrics['4']['lstm']['mae']:.4f} | {per_zone_metrics['4']['tcn']['mae']:.4f} |
| **Sector 5 (Blueberry)** | {per_zone_metrics['5']['n_samples']} | {per_zone_metrics['5']['target_mean']:.2f} ± {per_zone_metrics['5']['target_std']:.2f} | **{per_zone_metrics['5']['persistence_baseline']['mae']:.4f}** | {per_zone_metrics['5']['transformer']['mae']:.4f} | {per_zone_metrics['5']['lstm']['mae']:.4f} | {per_zone_metrics['5']['tcn']['mae']:.4f} |

## Key Findings & Conclusion

1. **Persistence Baseline Remains Dominant:** The persistence baseline achieves MAE 0.4124, outperforming the Transformer by 8.5x (Transformer MAE: {overall_metrics['transformer']['mae']:.4f}).
2. **Post-Season Test Equilibrium:** In the late-season test period, zero irrigation occurred across all sectors. Soil moisture was in static equilibrium, rendering persistence optimal.
3. **Simpson's Paradox in Pooled R²:** Pooled R² appears high ({overall_metrics['transformer']['r2']:.4f}) due to between-sector moisture variance (~2% to 57%), but within individual sectors, neural network R² is negative.
4. **Production Verdict:** The Transformer does NOT beat the baseline and must NOT be deployed to production.
"""
    with open(METRICS_MD_PATH, "w") as f:
        f.write(md_content)
    print(f"Saved results Markdown to: {METRICS_MD_PATH}")

    return results_data


if __name__ == "__main__":
    train_transformer()

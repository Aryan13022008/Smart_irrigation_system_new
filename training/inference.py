"""
Smart Irrigation System - Unified Model Inference Backend
========================================================================
Loads trained models (Random Forest, LSTM, TCN, Transformer) and their 
corresponding metric files to provide live predictions and comparison tables 
for the Streamlit dashboard.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "training" / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import database
except ImportError:
    database = None


# ---------------------------------------------------------------------------
# PyTorch Model Architectures (matching training scripts)
# ---------------------------------------------------------------------------

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size=11, hidden_dim=64, num_layers=2, n_features=None):
        super().__init__()
        if n_features is not None:
            input_size = n_features
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, n_features=11, channels=(64, 64, 128), kernel_size=5, dropout=0.3, input_size=None):
        super().__init__()
        if input_size is not None:
            n_features = input_size
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_step = out[:, :, -1]
        return self.head(last_step)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features=11, d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.3, input_size=None):
        super().__init__()
        if input_size is not None:
            n_features = input_size
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


ACTIVE_MODEL_NAME = "Random Forest (11-Feature)"

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

ORDERED_13_FEATURES = [
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
    "month_cos",
]


# ---------------------------------------------------------------------------
# Helper Loaders & Inference Wrappers
# ---------------------------------------------------------------------------

def get_metrics(model_name=None):
    if model_name is None:
        model_name = ACTIVE_MODEL_NAME
    mapping = {
        "Random Forest (11-Feature)": "random_forest_11feat_metrics.json",
        "Random Forest (11-Feature Reduced)": "random_forest_11feat_metrics.json",
        "random_forest_11feat": "random_forest_11feat_metrics.json",
        "Random Forest": "random_forest_metrics.json",
        "Random Forest (13-Feature)": "random_forest_metrics.json",
        "LSTM": "lstm_metrics.json",
        "TCN": "tcn_metrics.json",
        "Transformer": "transformer_metrics.json",
    }
    filename = mapping.get(model_name)
    if not filename:
        return None
    
    path = MODELS_DIR / filename
    if not path.exists():
        # Fallback check for alternate names
        if model_name == "LSTM":
            alt_path = MODELS_DIR / "improved_lstm_metrics.json"
            if alt_path.exists():
                path = alt_path
        if not path.exists():
            return None
            
    with open(path, "r") as f:
        return json.load(f)


def get_model_trustworthiness(model_name=None, sensor_data=None):
    """
    Returns an evidence-based trustworthiness assessment and input-readiness status
    for the selected model based on independent testing, target validity, and hardware inputs.
    """
    if model_name is None:
        model_name = ACTIVE_MODEL_NAME
    if sensor_data is None:
        sensor_data = {}

    has_raw_soil = (
        sensor_data.get("soil1_raw") is not None or
        sensor_data.get("soil2_raw") is not None or
        (
            sensor_data.get("soil_moisture") is not None and
            (float(sensor_data.get("soil_moisture", 0)) > 100.0 or sensor_data.get("is_calibrated") is False)
        )
    )
    has_calibrated_soil = (
        sensor_data.get("soil_moisture_calibrated") is not None or
        (sensor_data.get("is_calibrated") is True and sensor_data.get("soil_moisture") is not None and 0 <= float(sensor_data.get("soil_moisture")) <= 100)
    )

    rf11_m = get_metrics("random_forest_11feat") or {}
    rf11_thresh = float(rf11_m.get("threshold_tuned", 0.3270985189000874))

    profiles = {
        "Random Forest (11-Feature)": {
            "category": "Experimental — Advisory Only",
            "badge": "🟡 Experimental — Advisory Only (11-Feature Reduced Model)",
            "summary": "Trained on real IoT sensors excluding unavailable physical soil temperature and soil humidity. Evaluated on an independent chronological test set of 5,999 samples (ROC-AUC: 0.6808, PR-AUC: 0.6817, Recall: 85.95%, Precision: 62.64% @ threshold 0.3271). Advisory only: relative capacitive probe moisture has not been validated against dataset scale.",
            "evaluation_type": "Independent Chronological Test Set (N=5,999 Real IoT Data)",
            "target": "Real Irrigation Actuation (Status: ON / OFF)",
            "architecture": "HistGradientBoostingClassifier (11 features)",
            "threshold": rf11_thresh,
            "metrics": {
                "Test ROC-AUC": "0.6808",
                "Test PR-AUC": "0.6817",
                "Test Accuracy": "64.51%",
                "Test Recall": "85.95%",
                "Test Precision": "62.64%",
                "Test F1": "0.7247",
            },
            "input_requirements": "11 environmental features (calibrated soil moisture 0-100%, DHT22 air temp & air humidity, weather wind/gust/pressure, 1h SQLite lags, cyclical month)",
            "input_readiness": (
                "Ready for Live Prediction" if has_calibrated_soil else
                ("Blocked: Soil Probe Calibration Required (raw ADC received; requires 0-100% moisture percentage)" if has_raw_soil else
                 "Hardware Disconnected (sensor data N/A)")
            ),
            "input_readiness_code": "READY" if has_calibrated_soil else ("BLOCKED_CALIBRATION" if has_raw_soil else "DISCONNECTED"),
            "warning": "Relative soil moisture from capacitive sensors has not been field-validated against dataset scale. Automatic pump actuation is strictly disabled; predictions are advisory only.",
        },
        "Random Forest": {
            "category": "Best-supported",
            "badge": "🟢 Best-supported (Evaluated on Real IoT Test Set)",
            "summary": "Trained on real field IoT sensors to predict irrigation pump actuation (Status: ON/OFF). Evaluated on an independent chronological test set of 5,999 samples (Test ROC-AUC: 0.7927, PR-AUC: 0.7712, Recall: 93.10% @ threshold 0.325).",
            "evaluation_type": "Independent Chronological Test Set (Real IoT Data)",
            "target": "Real Irrigation Actuation (Status: ON / OFF)",
            "architecture": "HistGradientBoostingClassifier (13 features)",
            "threshold": 0.3252,
            "metrics": {
                "Test ROC-AUC": "0.7927",
                "Test PR-AUC": "0.7712",
                "Test Accuracy": "70.70%",
                "Test Recall": "93.10%",
                "Test Precision": "66.44%",
                "Test F1": "0.7754",
            },
            "input_requirements": "13 environmental features (calibrated soil moisture 0-100%, DHT22 air temp & air humidity, weather wind/gust/pressure, cyclical month)",
            "input_readiness": (
                "Ready for Live Prediction" if has_calibrated_soil else
                ("Blocked: Soil Probe Calibration Required (raw ADC received; requires 0-100% moisture percentage)" if has_raw_soil else
                 "Hardware Disconnected (sensor data N/A)")
            ),
            "input_readiness_code": "READY" if has_calibrated_soil else ("BLOCKED_CALIBRATION" if has_raw_soil else "DISCONNECTED"),
        },
        "LSTM": {
            "category": "Not validated for irrigation",
            "badge": "🔴 Not validated for irrigation (Random Chance Test ROC-AUC: 0.5008)",
            "summary": "Trained on synthetic spatial volatility across 11 research station plots, NOT crop irrigation demand. Independent test ROC-AUC is 0.5008 (equivalent to random guessing), predicting positive on 99.98% of cases at tuned threshold. Scaler was not saved, and inputs require 11 plot channels unavailable on single-probe hardware.",
            "evaluation_type": "Test Set (Synthetic Volatility Target)",
            "target": "Synthetic Plot Volatility (std > 60th percentile across 11 plots)",
            "architecture": "ImprovedLSTMClassifier (input_size=11, 2 layers, 64 hidden)",
            "threshold": 0.3802,
            "metrics": {
                "Test ROC-AUC": "0.5008",
                "Test PR-AUC": "0.5440",
                "Test Accuracy": "54.38%",
                "Test Recall": "99.97%",
                "Test Precision": "54.38%",
                "Test F1": "0.7044",
            },
            "input_requirements": "24-hour sequence window of 11 historical station plot irrigation depth channels (C535CA, C55D7C, ...)",
            "input_readiness": "Incompatible: Requires 11 historical research plot sequence channels; ESP32 provides single-probe telemetry.",
            "input_readiness_code": "INCOMPATIBLE_CHANNELS",
        },
        "TCN": {
            "category": "Not validated for irrigation",
            "badge": "🔴 Not validated for irrigation (Validation Only on Synthetic Target)",
            "summary": "Trained on synthetic spatial volatility across 11 research station plots, NOT crop irrigation demand. High reported ROC-AUC (0.9107) is validation-only from a random split with potential temporal leakage; no independent test evaluation exists. Feature scaler was not saved, and inputs require 11 station channels unavailable on single-probe hardware.",
            "evaluation_type": "Validation Split Only (Synthetic Volatility Target)",
            "target": "Synthetic Plot Volatility (std > 60th percentile across 11 plots)",
            "architecture": "TCNClassifier (n_features=11, 3 residual blocks, kernel=5)",
            "threshold": 0.9800,
            "metrics": {
                "Validation ROC-AUC": "0.9107",
                "Validation PR-AUC": "0.8603",
                "Test Evaluation": "None Available (No test set)",
                "Optimal Threshold": "0.9800",
            },
            "input_requirements": "24-hour sequence window of 11 historical station plot irrigation depth channels (C535CA, C55D7C, ...)",
            "input_readiness": "Incompatible: Requires 11 historical research plot sequence channels; ESP32 provides single-probe telemetry.",
            "input_readiness_code": "INCOMPATIBLE_CHANNELS",
        },
        "Transformer": {
            "category": "Not validated for irrigation",
            "badge": "🔴 Not validated for irrigation (Validation Only on Synthetic Target)",
            "summary": "Trained on synthetic spatial volatility across 11 research station plots, NOT crop irrigation demand. High reported ROC-AUC (0.9538) is validation-only from a random split with potential temporal leakage; no independent test evaluation exists. Feature scaler was not saved, and inputs require 11 station channels unavailable on single-probe hardware.",
            "evaluation_type": "Validation Split Only (Synthetic Volatility Target)",
            "target": "Synthetic Plot Volatility (std > 60th percentile across 11 plots)",
            "architecture": "TransformerClassifier (n_features=11, d_model=32, 4 heads, 2 layers)",
            "threshold": 0.6001,
            "metrics": {
                "Validation ROC-AUC": "0.9538",
                "Validation PR-AUC": "0.9291",
                "Test Evaluation": "None Available (No test set)",
                "Optimal Threshold": "0.6001",
            },
            "input_requirements": "24-hour sequence window of 11 historical station plot irrigation depth channels (C535CA, C55D7C, ...)",
            "input_readiness": "Incompatible: Requires 11 historical research plot sequence channels; ESP32 provides single-probe telemetry.",
            "input_readiness_code": "INCOMPATIBLE_CHANNELS",
        },
    }

    # Aliases
    profiles["Random Forest (11-Feature Reduced)"] = profiles["Random Forest (11-Feature)"]
    profiles["random_forest_11feat"] = profiles["Random Forest (11-Feature)"]
    profiles["Random Forest (13-Feature)"] = profiles["Random Forest"]

    return profiles.get(model_name, {
        "category": "Experimental",
        "badge": "⚪ Unknown Model",
        "summary": "No verified evaluation profile found.",
        "evaluation_type": "N/A",
        "target": "Unknown",
        "architecture": "Unknown",
        "threshold": 0.5,
        "metrics": {},
        "input_requirements": "Unknown",
        "input_readiness": "Unknown",
        "input_readiness_code": "UNKNOWN",
    })


def get_model_details(model_name, sensor_data=None):
    """
    Returns full metadata, architecture, metrics, and trustworthiness for a model.
    """
    trust = get_model_trustworthiness(model_name, sensor_data)
    metrics = get_metrics(model_name) or {}
    return {
        "model_name": model_name,
        "trustworthiness": trust,
        "metrics_raw": metrics,
        "features": metrics.get("features", []),
        "threshold": trust["threshold"],
    }


def get_model_comparison_rows():
    rows = []
    models = ["Random Forest (11-Feature)", "Random Forest", "LSTM", "TCN", "Transformer"]
    for m in models:
        metrics = get_metrics(m)
        trust = get_model_trustworthiness(m)
        if metrics is not None:
            if m == "Random Forest (11-Feature)":
                tuned = metrics.get("test_at_tuned_threshold", {})
                roc = tuned.get("roc_auc", 0.6808)
                pr = tuned.get("pr_auc", 0.6817)
                thresh = metrics.get("threshold_tuned", 0.3271)
                rows.append({
                    "Model": "Random Forest (11-Feature)",
                    "Architecture": "HistGradientBoosting (11-Feat)",
                    "Target Task": "Real Irrigation Demand (ON/OFF)",
                    "Evaluation Split": "Test Set (N=5,999 Real IoT)",
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Trustworthiness": trust["category"]
                })
            elif m == "Random Forest":
                tuned = metrics.get("test_at_tuned_threshold", {})
                roc = tuned.get("roc_auc", 0)
                pr = tuned.get("pr_auc", 0)
                thresh = metrics.get("threshold_tuned", 0.325)
                rows.append({
                    "Model": "Random Forest (13-Feat Baseline)",
                    "Architecture": "HistGradientBoosting (13-Feat)",
                    "Target Task": "Real Irrigation Demand (ON/OFF)",
                    "Evaluation Split": "Test Set (Real IoT Data)",
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Trustworthiness": "Offline Reference"
                })
            elif m == "LSTM":
                tuned = metrics.get("test_at_tuned_threshold", {})
                roc = tuned.get("roc_auc", 0)
                pr = tuned.get("pr_auc", 0)
                thresh = metrics.get("threshold_tuned", 0.380)
                rows.append({
                    "Model": m,
                    "Architecture": "Improved LSTM",
                    "Target Task": "Synthetic Plot Volatility",
                    "Evaluation Split": "Test Set (Station Target)",
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Trustworthiness": trust["category"]
                })
            else:  # TCN, Transformer
                roc = metrics.get("validation_roc_auc", 0)
                pr = metrics.get("validation_pr_auc", 0)
                thresh = metrics.get("optimal_threshold", 0.5)
                arch = "Temporal ConvNet" if m == "TCN" else "Transformer Encoder"
                rows.append({
                    "Model": m,
                    "Architecture": arch,
                    "Target Task": "Synthetic Plot Volatility",
                    "Evaluation Split": "Validation Only (No Test Set)",
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Trustworthiness": trust["category"]
                })
        else:
            rows.append({
                "Model": m,
                "Architecture": "N/A",
                "Target Task": "N/A",
                "Evaluation Split": "N/A",
                "ROC-AUC": "N/A",
                "PR-AUC": "N/A",
                "Tuned Threshold": "N/A",
                "Trustworthiness": "Not Trained"
            })
    return rows


def audit_feature_readiness(sensor_data=None, weather_data=None, db_path=None, model_name=None):
    """
    Audits live-input readiness against the model schema.
    Defaults to the active 11-feature Random Forest schema (random_forest_11feat.pkl).
    """
    if model_name is None:
        model_name = ACTIVE_MODEL_NAME
    if sensor_data is None:
        sensor_data = {}
    if weather_data is None:
        weather_data = {}

    if model_name in ("Random Forest (11-Feature)", "Random Forest (11-Feature Reduced)", "random_forest_11feat"):
        expected_features = list(ORDERED_11_FEATURES)
    elif model_name in ("Random Forest", "Random Forest (13-Feature)"):
        expected_features = list(ORDERED_13_FEATURES)
    else:
        m = get_metrics(model_name) or {}
        expected_features = m.get("features", list(ORDERED_11_FEATURES))

    # Query database for history span and 1-hour lag reading
    lag_reading = None
    history_span_sec = 0.0
    history_count = 0
    calibrations = {"probe1": {"calibrated": False}, "probe2": {"calibrated": False}}
    if database is not None:
        try:
            curr_ref_time = sensor_data.get("recorded_at")
            _, _, history_count, history_span_sec = database.get_history_span(db_path=db_path, continuous=True)
            lag_reading = database.get_lag_reading(
                target_seconds_ago=3600, tolerance_seconds=900, current_time=curr_ref_time, db_path=db_path, require_continuous=True
            )
            calibrations = database.get_all_calibrations(db_path=db_path)
        except Exception:
            pass

    # Check soil probe calibration status
    probe1_cal = calibrations.get("probe1", {})
    probe2_cal = calibrations.get("probe2", {})
    probe1_is_cal = probe1_cal.get("calibrated", False)
    probe2_is_cal = probe2_cal.get("calibrated", False)
    any_probe_cal = probe1_is_cal or probe2_is_cal

    # Evaluate current soil moisture
    current_soil_m = None
    soil_source = "ESP32 Soil Probes"
    soil_prov = "Unavailable (Hardware disconnected)"
    soil_ready = False
    soil_limit = "Sensor disconnected or N/A."

    if sensor_data.get("soil_moisture_calibrated") is not None:
        current_soil_m = float(sensor_data["soil_moisture_calibrated"])
        soil_prov = "Measured (Calibrated)"
        soil_source = "ESP32 Soil Probes (Calibrated)"
        soil_ready = True
        soil_limit = None
    elif sensor_data.get("is_calibrated") is True and sensor_data.get("soil_moisture") is not None:
        val = float(sensor_data["soil_moisture"])
        if val <= 100.0:
            current_soil_m = val
            soil_prov = "Measured (Calibrated)"
            soil_source = "ESP32 Soil Probes (Calibrated)"
            soil_ready = True
            soil_limit = None
        else:
            soil_prov = "Unavailable (Uncalibrated raw ADC)"
            soil_source = "ESP32 Soil Probes (Raw ADC)"
            soil_ready = False
            soil_limit = f"Soil moisture reading ({val:.0f}) exceeds 100%; treated as raw ADC. Calibration required."
    elif any_probe_cal and (sensor_data.get("soil1_raw") is not None or sensor_data.get("soil2_raw") is not None):
        vals = []
        if probe1_is_cal and sensor_data.get("soil1_raw") is not None and database is not None:
            m1 = database.calculate_calibrated_moisture(sensor_data["soil1_raw"], probe1_cal["dry_adc"], probe1_cal["wet_adc"])
            if m1 is not None:
                vals.append(m1)
        if probe2_is_cal and sensor_data.get("soil2_raw") is not None and database is not None:
            m2 = database.calculate_calibrated_moisture(sensor_data["soil2_raw"], probe2_cal["dry_adc"], probe2_cal["wet_adc"])
            if m2 is not None:
                vals.append(m2)
        if vals:
            current_soil_m = sum(vals) / len(vals)
            soil_prov = "Measured (Calibrated)"
            soil_source = "ESP32 Soil Probes (Calibrated from SQLite references)"
            soil_ready = True
            soil_limit = None
        else:
            soil_prov = "Observed (Raw ADC - Uncalibrated)"
            soil_source = "ESP32 Soil Probes (Raw ADC)"
            soil_ready = False
            soil_limit = "Raw ADC counts received from uncalibrated probe."
    elif sensor_data.get("soil1_raw") is not None or sensor_data.get("soil2_raw") is not None or sensor_data.get("soil_moisture") is not None:
        p1 = sensor_data.get("soil1_raw")
        p2 = sensor_data.get("soil2_raw")
        raw_sm = sensor_data.get("soil_moisture")
        probe_parts = []
        if p1 is not None:
            probe_parts.append(f"Probe 1: {float(p1):.0f} ADC")
        if p2 is not None:
            probe_parts.append(f"Probe 2: {float(p2):.0f} ADC")
        if not probe_parts and raw_sm is not None:
            probe_parts.append(f"Raw ADC: {float(raw_sm):.0f}")
        probes_str = ", ".join(probe_parts)
        soil_prov = "Observed (Raw ADC - Uncalibrated)"
        soil_source = "ESP32 Soil Probes (Raw ADC)"
        soil_ready = False
        soil_limit = f"Live reading ({probes_str}) received. Probes require measured dry/wet ADC calibration references."

    # Evaluate air temperature (feature: 'temperature')
    air_temp_val = None
    air_temp_source = "ESP32 DHT22 / Weather API"
    air_temp_prov = "Unavailable"
    air_temp_ready = False
    air_temp_limit = "No air temperature reading available."
    if sensor_data.get("temperature") is not None:
        air_temp_val = float(sensor_data["temperature"])
        air_temp_source = "ESP32 DHT22"
        air_temp_prov = "Measured (DHT22 Air Temp)"
        air_temp_ready = True
        air_temp_limit = None
    elif weather_data.get("temperature") is not None:
        air_temp_val = float(weather_data["temperature"])
        air_temp_source = "Open-Meteo API (2m Temp)"
        air_temp_prov = "Weather-derived (Open-Meteo)"
        air_temp_ready = True
        air_temp_limit = "Fallback to Open-Meteo current 2m temperature."

    # Evaluate air humidity (feature: 'air_humidity')
    air_hum_val = None
    air_hum_source = "ESP32 DHT22 / Weather API"
    air_hum_prov = "Unavailable"
    air_hum_ready = False
    air_hum_limit = "No air humidity reading available."
    if sensor_data.get("humidity") is not None:
        air_hum_val = float(sensor_data["humidity"])
        air_hum_source = "ESP32 DHT22"
        air_hum_prov = "Measured (DHT22 Air Humidity)"
        air_hum_ready = True
        air_hum_limit = None
    elif weather_data.get("humidity") is not None:
        air_hum_val = float(weather_data["humidity"])
        air_hum_source = "Open-Meteo API (2m Relative Humidity)"
        air_hum_prov = "Weather-derived (Open-Meteo)"
        air_hum_ready = True
        air_hum_limit = "Fallback to Open-Meteo current 2m humidity."

    # Evaluate wind speed (feature: 'wind_speed')
    wind_spd_val = None
    wind_spd_ready = False
    wind_spd_limit = "Weather wind speed unavailable."
    if weather_data.get("wind_speed") is not None:
        wind_spd_val = float(weather_data["wind_speed"]) * 3.6
        wind_spd_ready = True
        wind_spd_limit = None

    # Evaluate wind gust (feature: 'wind_gust')
    wind_gst_val = None
    wind_gst_ready = False
    wind_gst_limit = "Weather wind gust unavailable."
    if weather_data.get("wind_gust") is not None:
        wind_gst_val = float(weather_data["wind_gust"]) * 3.6
        wind_gst_ready = True
        wind_gst_limit = None

    # Evaluate pressure (feature: 'pressure')
    press_val = None
    press_ready = False
    press_limit = "Surface pressure unavailable."
    if weather_data.get("pressure") is not None:
        press_val = float(weather_data["pressure"]) / 10.0
        press_ready = True
        press_limit = None

    # Evaluate lag features from SQLite
    prev_soil_m = None
    prev_soil_ready = False
    prev_soil_prov = "Unavailable (Insufficient history span < 45m)"
    prev_soil_limit = f"Requires >= 45m continuous history in SQLite (currently {history_span_sec/60:.1f}m)."
    if lag_reading is not None and any_probe_cal and database is not None and history_span_sec >= 2700:
        lag_vals = []
        if probe1_is_cal and lag_reading.get("soil1_raw") is not None:
            m1 = database.calculate_calibrated_moisture(lag_reading["soil1_raw"], probe1_cal["dry_adc"], probe1_cal["wet_adc"])
            if m1 is not None:
                lag_vals.append(m1)
        if probe2_is_cal and lag_reading.get("soil2_raw") is not None:
            m2 = database.calculate_calibrated_moisture(lag_reading["soil2_raw"], probe2_cal["dry_adc"], probe2_cal["wet_adc"])
            if m2 is not None:
                lag_vals.append(m2)
        if lag_vals:
            prev_soil_m = sum(lag_vals) / len(lag_vals)
            prev_soil_ready = True
            prev_soil_prov = "Historically calculated (1-hour lag from SQLite)"
            prev_soil_limit = None
    elif lag_reading is not None and history_span_sec >= 2700:
        prev_soil_prov = "Unavailable (Historical probe uncalibrated)"
        prev_soil_limit = "Lag reading found in SQLite, but probe dry/wet references are uncalibrated."

    # Soil moisture change
    change_soil_m = None
    change_soil_ready = False
    change_soil_prov = "Unavailable (Requires current & 1h lag)"
    change_soil_limit = "Missing current or 1-hour lag calibrated soil moisture."
    if current_soil_m is not None and prev_soil_m is not None:
        change_soil_m = current_soil_m - prev_soil_m
        change_soil_ready = True
        change_soil_prov = "Historically calculated (Δ = current - 1h lag)"
        change_soil_limit = None

    # Previous temperature
    prev_temp_val = None
    prev_temp_ready = False
    prev_temp_prov = "Unavailable (Insufficient history span < 45m)"
    prev_temp_limit = f"Requires >= 45m continuous history in SQLite (currently {history_span_sec/60:.1f}m)."
    if lag_reading is not None and lag_reading.get("temperature_c") is not None and history_span_sec >= 2700:
        prev_temp_val = float(lag_reading["temperature_c"])
        prev_temp_ready = True
        prev_temp_prov = "Historically calculated (1-hour lag DHT22 temp)"
        prev_temp_limit = None

    # Cyclical astronomical clock features
    now = datetime.now()
    m_sin = float(np.sin(2 * np.pi * now.month / 12))
    m_cos = float(np.cos(2 * np.pi * now.month / 12))

    # Construct the exact 13 feature dictionary items
    if current_soil_m is not None:
        soil_display_val = f"{current_soil_m:.1f}%"
    elif sensor_data.get("soil1_raw") is not None or sensor_data.get("soil2_raw") is not None:
        p1 = sensor_data.get("soil1_raw")
        p2 = sensor_data.get("soil2_raw")
        p_strs = []
        if p1 is not None:
            p_strs.append(f"P1: {float(p1):.0f}")
        if p2 is not None:
            p_strs.append(f"P2: {float(p2):.0f}")
        soil_display_val = f"Raw ({'/'.join(p_strs)} ADC)"
    elif sensor_data.get("soil_moisture") is not None:
        soil_display_val = f"Raw ({float(sensor_data['soil_moisture']):.0f} ADC)"
    else:
        soil_display_val = "N/A"

    feature_meta = {
        "soil_moisture": {
            "units": "% (0–100)",
            "value": soil_display_val,
            "source": soil_source,
            "provenance": soil_prov,
            "is_ready": soil_ready,
            "limitation": soil_limit or "Ready: calibrated live probe measurement."
        },
        "soil_temperature": {
            "units": "°C",
            "value": "N/A",
            "source": "Soil Temp Sensor (Not Installed)",
            "provenance": "Unavailable (Sensor not installed)",
            "is_ready": False,
            "limitation": "Hardware sensor not installed. Ambient air temperature cannot substitute for root zone temperature."
        },
        "soil_humidity": {
            "units": "%",
            "value": "N/A",
            "source": "Soil Humidity Sensor (Not Installed)",
            "provenance": "Unavailable (Sensor not installed)",
            "is_ready": False,
            "limitation": "Hardware sensor not installed (trained as distinct feature from soil moisture in IoT dataset)."
        },
        "temperature": {
            "units": "°C",
            "value": f"{air_temp_val:.1f}°C" if air_temp_val is not None else "N/A",
            "source": air_temp_source,
            "provenance": air_temp_prov,
            "is_ready": air_temp_ready,
            "limitation": air_temp_limit or "Ready: ambient air temperature available."
        },
        "wind_speed": {
            "units": "km/h",
            "value": f"{wind_spd_val:.1f} km/h" if wind_spd_val is not None else "N/A",
            "source": "Open-Meteo API",
            "provenance": "Weather-derived (Converted m/s to km/h)" if wind_spd_ready else "Unavailable",
            "is_ready": wind_spd_ready,
            "limitation": wind_spd_limit or "Ready: weather observation available."
        },
        "air_humidity": {
            "units": "%",
            "value": f"{air_hum_val:.1f}%" if air_hum_val is not None else "N/A",
            "source": air_hum_source,
            "provenance": air_hum_prov,
            "is_ready": air_hum_ready,
            "limitation": air_hum_limit or "Ready: ambient air humidity available."
        },
        "wind_gust": {
            "units": "km/h",
            "value": f"{wind_gst_val:.1f} km/h" if wind_gst_val is not None else "N/A",
            "source": "Open-Meteo API",
            "provenance": "Weather-derived (Converted m/s to km/h)" if wind_gst_ready else "Unavailable",
            "is_ready": wind_gst_ready,
            "limitation": wind_gst_limit or "Ready: weather gust observation available."
        },
        "pressure": {
            "units": "kPa",
            "value": f"{press_val:.2f} kPa" if press_val is not None else "N/A",
            "source": "Open-Meteo API",
            "provenance": "Weather-derived (Converted hPa to kPa)" if press_ready else "Unavailable",
            "is_ready": press_ready,
            "limitation": press_limit or "Ready: surface pressure available."
        },
        "soil_moisture_previous": {
            "units": "%",
            "value": f"{prev_soil_m:.1f}%" if prev_soil_m is not None else "N/A",
            "source": "SQLite History (~1h lag)",
            "provenance": prev_soil_prov,
            "is_ready": prev_soil_ready,
            "limitation": prev_soil_limit or "Ready: genuine reading located in 45–75 min lag window."
        },
        "soil_moisture_change": {
            "units": "%",
            "value": f"{change_soil_m:+.1f}%" if change_soil_m is not None else "N/A",
            "source": "SQLite History Difference",
            "provenance": change_soil_prov,
            "is_ready": change_soil_ready,
            "limitation": change_soil_limit or "Ready: rate of change calculated from genuine lag."
        },
        "temp_previous": {
            "units": "°C",
            "value": f"{prev_temp_val:.1f}°C" if prev_temp_val is not None else "N/A",
            "source": "SQLite History (~1h lag)",
            "provenance": prev_temp_prov,
            "is_ready": prev_temp_ready,
            "limitation": prev_temp_limit or "Ready: genuine temperature located in 45–75 min lag window."
        },
        "month_sin": {
            "units": "unitless [-1, 1]",
            "value": f"{m_sin:.3f}",
            "source": "System Clock",
            "provenance": "Clock-derived (Astronomical cyclical sine)",
            "is_ready": True,
            "limitation": "Ready: computed directly from system date/month."
        },
        "month_cos": {
            "units": "unitless [-1, 1]",
            "value": f"{m_cos:.3f}",
            "source": "System Clock",
            "provenance": "Clock-derived (Astronomical cyclical cosine)",
            "is_ready": True,
            "limitation": "Ready: computed directly from system date/month."
        },
    }

    feature_rows = []
    blockers = []
    for f in expected_features:
        info = feature_meta.get(f, {
            "units": "Unknown",
            "value": "N/A",
            "source": "Unknown",
            "provenance": "Unavailable",
            "is_ready": False,
            "limitation": f"Feature '{f}' not mapped."
        })
        row = {
            "Feature": f,
            "Units": info["units"],
            "Live Value": info["value"],
            "Source": info["source"],
            "Provenance": info["provenance"],
            "Ready": "✅ Ready" if info["is_ready"] else "❌ Blocked",
            "Limitation / Details": info["limitation"],
            "is_ready": info["is_ready"]
        }
        feature_rows.append(row)
        if not info["is_ready"]:
            blockers.append(f"{f}: {info['limitation']}")

    ready_count = sum(1 for r in feature_rows if r["is_ready"])
    is_live_ready = (ready_count == len(expected_features))

    return {
        "features": expected_features,
        "feature_rows": feature_rows,
        "ready_count": ready_count,
        "total_features": len(expected_features),
        "is_live_ready": is_live_ready,
        "blockers": blockers,
        "probe1_calibrated": probe1_is_cal,
        "probe2_calibrated": probe2_is_cal,
        "history_span_minutes": round(history_span_sec / 60.0, 1),
        "history_count": history_count,
    }


def predict(model_name=None, sensor_data=None, weather_data=None, sequence_data=None, enforce_complete_readiness=False, db_path=None):
    """
    Executes inference using the requested model.
    Defaults to the active 11-feature Random Forest model (random_forest_11feat.pkl).

    Parameters:
    - model_name: 'Random Forest (11-Feature)', 'Random Forest', 'LSTM', 'TCN', or 'Transformer'
    - sensor_data: dict containing live or calibrated sensor values
    - weather_data: dict containing live weather observations
    - sequence_data: optional array/tensor of shape (batch, 24, 11) for DL models
    - enforce_complete_readiness: if True, requires all expected features to be genuinely observed before predicting
    - db_path: optional SQLite database path for testing
    """
    if model_name is None:
        model_name = ACTIVE_MODEL_NAME
    if sensor_data is None:
        sensor_data = {}
    if weather_data is None:
        weather_data = {}

    device = torch.device('cpu')

    # -----------------------------------------------------------------------
    # 1. 11-Feature Reduced Model (Active Production Model)
    # -----------------------------------------------------------------------
    if model_name in ("Random Forest (11-Feature)", "Random Forest (11-Feature Reduced)", "random_forest_11feat"):
        metrics = get_metrics("random_forest_11feat") or {}
        rf_threshold = float(metrics.get("threshold_tuned", 0.3270985189000874))

        # Check full live readiness audit if enforcement requested
        if enforce_complete_readiness:
            readiness = audit_feature_readiness(sensor_data, weather_data, db_path=db_path, model_name=model_name)
            if not readiness["is_live_ready"]:
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Live prediction is withheld as N/A: {len(readiness['blockers'])} feature readiness blocker(s) present "
                        f"({readiness['ready_count']}/{readiness['total_features']} features ready). "
                        "Active blockers include uncalibrated probes or insufficient historical lag data. "
                        "In accordance with data integrity safeguards, training medians will not be silently substituted as observed physical live inputs."
                    ),
                    "readiness": readiness,
                    "model": model_name,
                }

        model_path = MODELS_DIR / "random_forest_11feat.pkl"
        if not model_path.exists():
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "11-feature model file (random_forest_11feat.pkl) not found.",
                "model": model_name,
            }

        # Check sensor data availability
        has_moisture_key = (
            sensor_data.get("soil_moisture") is not None or
            sensor_data.get("soil_moisture_calibrated") is not None
        )
        if not has_moisture_key and sensor_data.get("soil1_raw") is None and sensor_data.get("soil2_raw") is None:
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "Hardware not connected: live sensor data is N/A. Cannot run Random Forest prediction.",
                "model": model_name,
            }

        # Check soil moisture calibration scale
        soil_m = None
        if sensor_data.get("soil_moisture_calibrated") is not None:
            soil_m = float(sensor_data["soil_moisture_calibrated"])
        elif sensor_data.get("is_calibrated") is True and sensor_data.get("soil_moisture") is not None:
            soil_m = float(sensor_data["soil_moisture"])
        elif sensor_data.get("soil_moisture") is not None:
            val = float(sensor_data["soil_moisture"])
            if val > 100.0 or sensor_data.get("is_calibrated") is False:
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Soil sensor reading is raw ADC ({val:.0f}), not calibrated soil moisture percentage (0-100%). "
                        "Decision is retained as N/A until probe calibration is configured."
                    ),
                    "model": model_name,
                }
            soil_m = val

        if soil_m is None:
            if sensor_data.get("soil1_raw") is not None or sensor_data.get("soil2_raw") is not None:
                p1 = sensor_data.get("soil1_raw")
                p2 = sensor_data.get("soil2_raw")
                p_parts = []
                if p1 is not None:
                    p_parts.append(f"Probe 1: {float(p1):.0f}")
                if p2 is not None:
                    p_parts.append(f"Probe 2: {float(p2):.0f}")
                p_str = ", ".join(p_parts)
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Soil sensor reading is raw ADC ({p_str}), not calibrated soil moisture percentage (0-100%). "
                        "Decision is retained as N/A until probe calibration is configured."
                    ),
                    "model": model_name,
                }
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "Calibrated soil moisture reading is N/A. Decision is retained as N/A.",
                "model": model_name,
            }

        try:
            rf_model = joblib.load(model_path)
            medians = metrics.get("feature_medians", {})
            features = metrics.get("features", ORDERED_11_FEATURES)

            input_data = dict(medians)
            input_data["soil_moisture"] = soil_m

            # Air temperature: prefer ESP32 DHT22, fallback to Open-Meteo
            if sensor_data.get("temperature") is not None:
                input_data["temperature"] = float(sensor_data["temperature"])
            elif weather_data.get("temperature") is not None:
                input_data["temperature"] = float(weather_data["temperature"])

            # Air humidity: mapped to model's 'air_humidity'
            if sensor_data.get("humidity") is not None:
                input_data["air_humidity"] = float(sensor_data["humidity"])
            elif weather_data.get("humidity") is not None:
                input_data["air_humidity"] = float(weather_data["humidity"])

            # Wind speed: from weather_data (m/s -> km/h)
            if weather_data.get("wind_speed") is not None:
                input_data["wind_speed"] = float(weather_data["wind_speed"]) * 3.6

            # Wind gust: from weather_data (m/s -> km/h)
            if weather_data.get("wind_gust") is not None:
                input_data["wind_gust"] = float(weather_data["wind_gust"]) * 3.6

            # Pressure: from weather_data (hPa -> kPa)
            if weather_data.get("pressure") is not None:
                input_data["pressure"] = float(weather_data["pressure"]) / 10.0

            # Lags from database if available
            if database is not None:
                try:
                    curr_ref_time = sensor_data.get("recorded_at")
                    lag_r = database.get_lag_reading(
                        target_seconds_ago=3600, tolerance_seconds=900, current_time=curr_ref_time, db_path=db_path, require_continuous=True
                    )
                    cals = database.get_all_calibrations(db_path=db_path)
                    p1_cal = cals.get("probe1", {})
                    p2_cal = cals.get("probe2", {})
                    if lag_r is not None and (p1_cal.get("calibrated") or p2_cal.get("calibrated")):
                        l_vals = []
                        if p1_cal.get("calibrated") and lag_r.get("soil1_raw") is not None:
                            m1 = database.calculate_calibrated_moisture(lag_r["soil1_raw"], p1_cal["dry_adc"], p1_cal["wet_adc"])
                            if m1 is not None:
                                l_vals.append(m1)
                        if p2_cal.get("calibrated") and lag_r.get("soil2_raw") is not None:
                            m2 = database.calculate_calibrated_moisture(lag_r["soil2_raw"], p2_cal["dry_adc"], p2_cal["wet_adc"])
                            if m2 is not None:
                                l_vals.append(m2)
                        if l_vals:
                            prev_sm = sum(l_vals) / len(l_vals)
                            input_data["soil_moisture_previous"] = prev_sm
                            input_data["soil_moisture_change"] = soil_m - prev_sm
                    if lag_r is not None and lag_r.get("temperature_c") is not None:
                        input_data["temp_previous"] = float(lag_r["temperature_c"])
                except Exception:
                    pass

            # Cyclical month features from current date
            now = datetime.now()
            input_data["month_sin"] = float(np.sin(2 * np.pi * now.month / 12))
            input_data["month_cos"] = float(np.cos(2 * np.pi * now.month / 12))

            row = [input_data.get(f, medians.get(f, 0.0)) for f in features]
            X_live = pd.DataFrame([row], columns=features)

            proba = float(rf_model.predict_proba(X_live)[0, 1])
            threshold = rf_threshold
            required = bool(proba >= threshold)
            return {
                "required": required,
                "confidence": proba,
                "threshold": threshold,
                "message": (
                    f"11-Feature Random Forest prediction complete (P(irrigate)={proba:.1%}, threshold={threshold:.4f}). "
                    "[Experimental — Advisory Only: unvalidated capacitive probe scale]"
                ),
                "model": model_name,
                "readiness": audit_feature_readiness(sensor_data, weather_data, db_path=db_path, model_name=model_name),
            }
        except Exception as e:
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": f"Error running 11-Feature Random Forest inference: {str(e)}",
                "model": model_name,
            }

    # -----------------------------------------------------------------------
    # 2. Random Forest Baseline (HistGradientBoosting on 13 IoT Features - Offline Reference)
    # -----------------------------------------------------------------------
    if model_name in ("Random Forest", "Random Forest (13-Feature)"):
        metrics = get_metrics("Random Forest") or {}
        rf_threshold = float(metrics.get("threshold_tuned", 0.3252))

        # Check full live readiness audit if enforcement requested
        if enforce_complete_readiness:
            readiness = audit_feature_readiness(sensor_data, weather_data, db_path=db_path)
            if not readiness["is_live_ready"]:
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Live prediction is withheld as N/A: {len(readiness['blockers'])} feature readiness blocker(s) present "
                        f"({readiness['ready_count']}/{readiness['total_features']} features ready). "
                        "Active blockers include uncalibrated probes, missing physical soil sensors, or insufficient historical lag data. "
                        "In accordance with data integrity safeguards, training medians will not be silently substituted as observed physical live inputs."
                    ),
                    "readiness": readiness,
                }

        model_path = MODELS_DIR / "random_forest.pkl"
        if not model_path.exists():
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "Random Forest model file (random_forest.pkl) not found."
            }

        # Check sensor data availability
        has_moisture_key = (
            sensor_data.get("soil_moisture") is not None or
            sensor_data.get("soil_moisture_calibrated") is not None
        )
        if not has_moisture_key and sensor_data.get("soil1_raw") is None and sensor_data.get("soil2_raw") is None:
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "Hardware not connected: live sensor data is N/A. Cannot run Random Forest prediction."
            }

        # Check soil moisture calibration scale
        soil_m = None
        if sensor_data.get("soil_moisture_calibrated") is not None:
            soil_m = float(sensor_data["soil_moisture_calibrated"])
        elif sensor_data.get("is_calibrated") is True and sensor_data.get("soil_moisture") is not None:
            soil_m = float(sensor_data["soil_moisture"])
        elif sensor_data.get("soil_moisture") is not None:
            val = float(sensor_data["soil_moisture"])
            # If ADC raw reading (> 100 on 12-bit ADC) or marked uncalibrated
            if val > 100.0 or sensor_data.get("is_calibrated") is False:
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Soil sensor reading is raw ADC ({val:.0f}), not calibrated soil moisture percentage (0-100%). "
                        "Decision is retained as N/A until probe calibration is configured."
                    )
                }
            soil_m = val

        if soil_m is None:
            if sensor_data.get("soil1_raw") is not None or sensor_data.get("soil2_raw") is not None:
                p1 = sensor_data.get("soil1_raw")
                p2 = sensor_data.get("soil2_raw")
                p_parts = []
                if p1 is not None:
                    p_parts.append(f"Probe 1: {float(p1):.0f}")
                if p2 is not None:
                    p_parts.append(f"Probe 2: {float(p2):.0f}")
                p_str = ", ".join(p_parts)
                return {
                    "required": None,
                    "confidence": None,
                    "threshold": rf_threshold,
                    "message": (
                        f"Soil sensor reading is raw ADC ({p_str}), not calibrated soil moisture percentage (0-100%). "
                        "Decision is retained as N/A until probe calibration is configured."
                    )
                }
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": "Calibrated soil moisture reading is N/A. Decision is retained as N/A."
            }

        try:
            rf_model = joblib.load(model_path)
            medians = metrics.get("feature_medians", {})
            features = metrics.get("features", list(medians.keys()))

            # Feature mapping with careful distinction between air_humidity and humidity
            input_data = dict(medians)
            input_data["soil_moisture"] = soil_m

            # Air temperature: prefer ESP32 DHT22, fallback to Open-Meteo
            if sensor_data.get("temperature") is not None:
                input_data["temperature"] = float(sensor_data["temperature"])
            elif weather_data.get("temperature") is not None:
                input_data["temperature"] = float(weather_data["temperature"])

            # Air humidity: mapped to model's 'air_humidity' (NOT 'humidity' or 'soil_humidity'!)
            if sensor_data.get("humidity") is not None:
                input_data["air_humidity"] = float(sensor_data["humidity"])
            elif weather_data.get("humidity") is not None:
                input_data["air_humidity"] = float(weather_data["humidity"])

            # Wind speed: from weather_data (Open-Meteo m/s -> km/h to match training scale)
            if weather_data.get("wind_speed") is not None:
                input_data["wind_speed"] = float(weather_data["wind_speed"]) * 3.6

            # Wind gust: from weather_data if present (m/s -> km/h)
            if weather_data.get("wind_gust") is not None:
                input_data["wind_gust"] = float(weather_data["wind_gust"]) * 3.6

            # Pressure: from weather_data if present (hPa -> kPa)
            if weather_data.get("pressure") is not None:
                input_data["pressure"] = float(weather_data["pressure"]) / 10.0

            # Cyclical month features from current date
            now = datetime.now()
            input_data["month_sin"] = float(np.sin(2 * np.pi * now.month / 12))
            input_data["month_cos"] = float(np.cos(2 * np.pi * now.month / 12))

            # Features not measured directly (soil_temperature, soil_humidity, soil_moisture_previous,
            # soil_moisture_change, temp_previous) legitimately fall back to training-set medians.
            row = [input_data.get(f, medians.get(f, 0.0)) for f in features]
            X_live = pd.DataFrame([row], columns=features)

            proba = float(rf_model.predict_proba(X_live)[0, 1])
            threshold = rf_threshold
            required = bool(proba >= threshold)
            return {
                "required": required,
                "confidence": proba,
                "threshold": threshold,
                "message": f"Random Forest prediction complete (P(irrigate)={proba:.1%}, threshold={threshold:.2f})."
            }
        except Exception as e:
            return {
                "required": None,
                "confidence": None,
                "threshold": rf_threshold,
                "message": f"Error running Random Forest inference: {str(e)}"
            }

    # -----------------------------------------------------------------------
    # 2. Deep Learning Models (LSTM, TCN, Transformer)
    # -----------------------------------------------------------------------
    file_map = {
        "LSTM": ("improved_lstm.pt", ImprovedLSTMClassifier),
        "TCN": ("tcn.pt", TCNClassifier),
        "Transformer": ("transformer.pt", TransformerClassifier)
    }

    if model_name not in file_map:
        return {
            "required": None,
            "confidence": None,
            "threshold": 0.5,
            "message": f"Unknown model type: {model_name}"
        }

    dl_metrics = get_metrics(model_name) or {}
    dl_threshold = float(dl_metrics.get("optimal_threshold", dl_metrics.get("threshold_tuned", 0.5)))

    filename, model_cls = file_map[model_name]
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        return {
            "required": None,
            "confidence": None,
            "threshold": dl_threshold,
            "message": f"{model_name} weights file ({filename}) not found."
        }

    # Check input availability: DL models require sequence_data of shape (batch, 24, 11)
    # matching the 11 historical station irrigation channels on which they were trained.
    if sequence_data is None:
        return {
            "required": None,
            "confidence": None,
            "threshold": dl_threshold,
            "message": (
                f"{model_name} was trained on 11 historical station irrigation plot channels "
                "(C535CA, C55D7C, ...), not live single-probe ESP32 telemetry. "
                "Inputs cannot be fabricated; decision is retained as N/A."
            )
        }

    try:
        model = model_cls(n_features=11).to(device)
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        model.eval()

        if isinstance(sequence_data, np.ndarray):
            seq_tensor = torch.tensor(sequence_data, dtype=torch.float32)
        elif isinstance(sequence_data, torch.Tensor):
            seq_tensor = sequence_data.float()
        else:
            seq_tensor = torch.tensor(sequence_data, dtype=torch.float32)

        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        seq_tensor = seq_tensor.to(device)

        with torch.no_grad():
            logit = model(seq_tensor)
            proba = float(torch.sigmoid(logit).squeeze().item())

        threshold = dl_threshold
        required = bool(proba >= threshold)
        return {
            "required": required,
            "confidence": proba,
            "threshold": threshold,
            "message": f"{model_name} sequence evaluation complete (P(irrigate)={proba:.1%}, threshold={threshold:.2f})."
        }
    except Exception as e:
        return {
            "required": None,
            "confidence": None,
            "threshold": dl_threshold,
            "message": f"Error running {model_name} inference: {str(e)}"
        }

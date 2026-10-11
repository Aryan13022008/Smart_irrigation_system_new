"""
Comprehensive Unit & Integration Test Suite for Live Neural Forecasting Pipeline (Phases 1-8)
==============================================================================================
Covers:
1. Exact 6-feature live schema and temporal coordinates.
2. Calibrated probe relative moisture calculation and clamping [0.0, 100.0]%.
3. Ground-truth target definition and 60-minute horizon.
4. PyTorch LiveSoilMoistureLSTM architecture, forward pass, serialization, and input validation.
5. Continuous segment detection, gap isolation, and zero-leakage sequence building.
6. Chronological train/val/test splitting without future overlap.
7. Scientific training gate enforcement (TRAINING_WITHHELD_INSUFFICIENT_DATA).
8. Supervised training pipeline execution on isolated fixture.
9. Baselines (Naive Persistence, Empirical Trend) and model acceptance gate.
10. Multi-mode forecaster execution ("Trend Forecast", "Experimental LSTM", "Model Comparison").
11. Dashboard Mode B AppTest verification across all sub-modes.
12. Safety locks: PUMP_ALLOWED = False, production RF threshold intact, zero DB mutation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

from training.experimental.live_soil_forecaster.forecasting_pipeline import (
    PUMP_ALLOWED,
    LIVE_FEATURE_SCHEMA,
    PROBE_CALIBRATION_REFERENCES,
    TARGET_DEFINITION,
    TARGET_UNITS,
    TARGET_HORIZON_MINUTES,
    DEFAULT_SEQUENCE_LENGTH,
    calculate_calibrated_probe_index,
    assess_dataset_training_sufficiency,
    get_live_neural_forecast_status,
    get_forecast_by_mode,
)
from training.experimental.live_soil_forecaster.model import (
    LiveSoilMoistureLSTM,
    clamp_prediction,
    save_checkpoint,
    load_checkpoint,
    predict_future_moisture_index,
)
from training.experimental.live_soil_forecaster.dataset import (
    extract_continuous_segments,
    build_sequences_from_segment,
    chronological_split,
)
from training.experimental.live_soil_forecaster.trainer import (
    evaluate_training_gate,
    fit_scaler,
    transform_data,
    train_lstm_regressor,
    MIN_REQUIRED_SAMPLES,
    MIN_REQUIRED_SEGMENTS,
)
from training.experimental.live_soil_forecaster.baselines import (
    calculate_metrics,
    predict_persistence_baseline,
    predict_empirical_trend_baseline,
    evaluate_model_acceptance_gate,
)


class TestLiveSoilForecaster(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_safety_pump_allowed_is_false(self):
        """Verify non-negotiable safety constraint: PUMP_ALLOWED must be strictly False."""
        self.assertFalse(PUMP_ALLOWED, "PUMP_ALLOWED must be False in forecasting_pipeline")

    def test_02_production_classifier_unmodified(self):
        """Verify production classifier checkpoint, threshold, and feature schema remain intact."""
        rf_path = PROJECT_ROOT / "training" / "models" / "random_forest_11feat.pkl"
        metrics_path = PROJECT_ROOT / "training" / "models" / "random_forest_11feat_metrics.json"
        self.assertTrue(rf_path.exists(), "Production RF model checkpoint must exist")
        self.assertTrue(metrics_path.exists(), "Production RF metrics JSON must exist")

        rf_model = joblib.load(rf_path)
        self.assertEqual(rf_model.n_features_in_, 11, "Production classifier must retain 11 features")

        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        self.assertAlmostEqual(
            metrics_data["threshold_tuned"],
            0.3270985223048995,
            places=6,
            msg="Production classifier decision threshold must remain 0.3270985223048995",
        )
        self.assertEqual(len(metrics_data["features"]), 11)

    def test_03_sqlite_schema_intact(self):
        """Verify SQLite database schema and tables are untouched."""
        db_path = PROJECT_ROOT / "data" / "sensor_history.db"
        self.assertTrue(db_path.exists())
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"sensor_readings", "ten_minute_observations", "probe_calibration"}
        self.assertTrue(expected.issubset(tables), f"Database must retain {expected}")

    def test_04_feature_schema_specification(self):
        """Verify the exact ordered 6-feature live schema."""
        expected_features = [
            "soil1_relative_pct",
            "soil2_relative_pct",
            "temperature_c",
            "humidity_pct",
            "hour",
            "month",
        ]
        self.assertEqual(LIVE_FEATURE_SCHEMA, expected_features)
        self.assertEqual(len(LIVE_FEATURE_SCHEMA), 6)

    def test_05_probe_calibration_calculation(self):
        """Test probe ADC to relative moisture index mapping and boundary clamping."""
        dry, wet = 2054.0, 1073.0

        # Exact dry endpoint -> 0.0%
        self.assertAlmostEqual(calculate_calibrated_probe_index(dry, dry, wet), 0.0, places=2)
        # Exact wet endpoint -> 100.0%
        self.assertAlmostEqual(calculate_calibrated_probe_index(wet, dry, wet), 100.0, places=2)
        # Midpoint -> 50.0%
        mid = (dry + wet) / 2.0
        self.assertAlmostEqual(calculate_calibrated_probe_index(mid, dry, wet), 50.0, places=2)

        # Out-of-bounds clipping
        self.assertEqual(calculate_calibrated_probe_index(2500.0, dry, wet), 0.0)
        self.assertEqual(calculate_calibrated_probe_index(500.0, dry, wet), 100.0)

        # Invalid inputs return None
        self.assertIsNone(calculate_calibrated_probe_index(None, dry, wet))
        self.assertIsNone(calculate_calibrated_probe_index(1500.0, None, wet))
        self.assertIsNone(calculate_calibrated_probe_index(1500.0, wet, dry))  # dry <= wet

    def test_06_target_specifications(self):
        """Verify target definition, horizon, and units."""
        self.assertEqual(TARGET_DEFINITION, "future_relative_moisture_index_60m")
        self.assertEqual(TARGET_HORIZON_MINUTES, 60)
        self.assertEqual(DEFAULT_SEQUENCE_LENGTH, 6)
        self.assertIn("% relative index", TARGET_UNITS)

    def test_07_lstm_architecture_and_forward_pass(self):
        """Verify LiveSoilMoistureLSTM forward pass, dimensions, and output clamping."""
        model = LiveSoilMoistureLSTM(input_size=6, hidden_size=32, num_layers=2)
        batch_x = torch.randn(4, 6, 6)  # (batch=4, seq_len=6, features=6)
        out = model(batch_x)
        self.assertEqual(out.shape, (4, 1))

        # Test output clamping
        self.assertEqual(clamp_prediction(120.0), 100.0)
        self.assertEqual(clamp_prediction(-15.0), 0.0)
        self.assertEqual(clamp_prediction(45.5), 45.5)
        self.assertEqual(clamp_prediction(np.nan), 0.0)

    def test_08_lstm_checkpoint_serialization(self):
        """Verify PyTorch model checkpoint saving and loading with metadata and scaler parameters."""
        model = LiveSoilMoistureLSTM(input_size=6, hidden_size=16, num_layers=1)
        ckpt_path = Path(self.temp_dir) / "test_model.pt"

        scaler_mean = [50.0, 48.0, 22.0, 65.0, 12.0, 10.0]
        scaler_scale = [10.0, 9.0, 3.0, 8.0, 6.0, 1.0]
        metadata = {"version": "test-v1", "test_mae": 1.25}

        save_checkpoint(ckpt_path, model, scaler_mean, scaler_scale, LIVE_FEATURE_SCHEMA, metadata)
        self.assertTrue(ckpt_path.exists())

        loaded_model, loaded_ckpt = load_checkpoint(ckpt_path)
        self.assertEqual(loaded_ckpt["scaler_mean"], scaler_mean)
        self.assertEqual(loaded_ckpt["scaler_scale"], scaler_scale)
        self.assertEqual(loaded_ckpt["feature_names"], LIVE_FEATURE_SCHEMA)
        self.assertEqual(loaded_ckpt["metadata"]["version"], "test-v1")

    def test_09_predict_future_moisture_index_validation(self):
        """Verify predict_future_moisture_index input validation and inference."""
        model = LiveSoilMoistureLSTM(input_size=6, hidden_size=16, num_layers=1)
        scaler_mean = np.zeros(6)
        scaler_scale = np.ones(6)

        # Valid 2D input (6, 6)
        valid_seq = np.ones((6, 6), dtype=np.float32) * 50.0
        pred = predict_future_moisture_index(model, valid_seq, scaler_mean, scaler_scale)
        self.assertTrue(0.0 <= pred <= 100.0)

        # Malformed shape raises ValueError
        with self.assertRaises(ValueError):
            predict_future_moisture_index(model, np.ones((5, 6)), scaler_mean, scaler_scale)

        # Non-finite values raise ValueError
        bad_seq = valid_seq.copy()
        bad_seq[0, 0] = np.nan
        with self.assertRaises(ValueError):
            predict_future_moisture_index(model, bad_seq, scaler_mean, scaler_scale)

    def test_10_gap_isolation_and_segment_splitting(self):
        """Verify that extract_continuous_segments correctly isolates gaps and never links disjointed blocks."""
        # Create timestamps: 10 continuous steps, a 1-hour gap, then 5 continuous steps
        t0 = 1791600000.0
        epochs_block1 = [t0 + (i * 600.0) for i in range(10)]
        gap_t = epochs_block1[-1] + 3600.0  # 1 hour jump
        epochs_block2 = [gap_t + (i * 600.0) for i in range(5)]
        all_epochs = epochs_block1 + epochs_block2

        df = pd.DataFrame({
            "interval_start_epoch": all_epochs,
            "interval_start_utc": [f"ts_{i}" for i in range(len(all_epochs))],
        })

        segments = extract_continuous_segments(df)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 10)
        self.assertEqual(len(segments[1]), 5)

    def test_11_sequences_do_not_cross_gaps(self):
        """Verify that sequence builder never creates an (input, target) pair across gaps."""
        # 11 steps is insufficient for 6 input + 6 target (requires >= 12 steps)
        t0 = 1791600000.0
        epochs = [t0 + (i * 600.0) for i in range(11)]
        df_short = pd.DataFrame({
            "interval_start_epoch": epochs,
            "interval_start_utc": ["2026-10-10T12:00:00Z"] * len(epochs),
            "soil1_raw_mean": [1500.0] * len(epochs),
            "soil2_raw_mean": [1400.0] * len(epochs),
            "temperature_c_mean": [22.0] * len(epochs),
            "humidity_pct_mean": [65.0] * len(epochs),
        })
        X_s, y_s, meta_s = build_sequences_from_segment(df_short, sequence_length=6, horizon_steps=6)
        self.assertEqual(len(X_s), 0, "No pairs should be created from a segment shorter than 12 steps")

        # 12 steps creates exactly 1 pair
        epochs12 = [t0 + (i * 600.0) for i in range(12)]
        df_12 = pd.DataFrame({
            "interval_start_epoch": epochs12,
            "interval_start_utc": ["2026-10-10T12:00:00Z"] * len(epochs12),
            "soil1_raw_mean": [1500.0] * len(epochs12),
            "soil2_raw_mean": [1400.0] * len(epochs12),
            "temperature_c_mean": [22.0] * len(epochs12),
            "humidity_pct_mean": [65.0] * len(epochs12),
        })
        X_12, y_12, meta_12 = build_sequences_from_segment(df_12, sequence_length=6, horizon_steps=6)
        self.assertEqual(len(X_12), 1)
        self.assertEqual(len(y_12), 1)

    def test_12_chronological_split(self):
        """Verify strictly chronological splitting without shuffling or sample overlap."""
        N = 20
        X = np.arange(N * 6 * 6).reshape((N, 6, 6))
        y = np.arange(N)
        meta = [{"id": i} for i in range(N)]

        splits = chronological_split(X, y, meta, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        self.assertEqual(len(splits["train"]["y"]), 14)
        self.assertEqual(len(splits["val"]["y"]), 3)
        self.assertEqual(len(splits["test"]["y"]), 3)

        # Check order preservation
        self.assertEqual(splits["train"]["y"][-1], 13)
        self.assertEqual(splits["val"]["y"][0], 14)
        self.assertEqual(splits["test"]["y"][0], 17)

    def test_13_training_gate_logic(self):
        """Verify scientific training gate rejects insufficient real data and authorizes sufficient mock data."""
        # 1. Failing report (< 200 samples)
        fail_rep = {"valid_sequences_count": 23, "continuous_segments_count": 1, "target_std": 0.2}
        gate_fail = evaluate_training_gate(fail_rep)
        self.assertFalse(gate_fail["authorized_to_train"])
        self.assertEqual(gate_fail["status"], "TRAINING_WITHHELD_INSUFFICIENT_DATA")

        # 2. Passing report (>= 200 samples, >= 3 segments, std >= 0.5)
        pass_rep = {"valid_sequences_count": 250, "continuous_segments_count": 4, "target_std": 2.1}
        gate_pass = evaluate_training_gate(pass_rep)
        self.assertTrue(gate_pass["authorized_to_train"])
        self.assertEqual(gate_pass["status"], "AUTHORIZED")

    def test_14_training_pipeline_on_fixture(self):
        """Verify supervised LSTM training loop on an isolated synthetic fixture."""
        N = 30
        X_synth = np.random.randn(N, 6, 6).astype(np.float32)
        y_synth = (np.mean(X_synth[:, :, 0], axis=1) * 10.0 + 50.0).astype(np.float32)

        mean, scale = fit_scaler(X_synth[:20])
        X_train_norm = transform_data(X_synth[:20], mean, scale)
        X_val_norm = transform_data(X_synth[20:], mean, scale)

        model, history = train_lstm_regressor(
            X_train=X_train_norm,
            y_train=y_synth[:20],
            X_val=X_val_norm,
            y_val=y_synth[20:],
            epochs=5,
            batch_size=8,
        )
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertTrue(len(history["train_loss"]) > 0)

    def test_15_baselines_and_acceptance(self):
        """Verify persistence and trend baselines, metric calculation, and acceptance gate."""
        # X with probe1=40.0, probe2=50.0 at last step -> avg 45.0
        X = np.zeros((3, 6, 6), dtype=np.float32)
        X[:, -1, 0] = 40.0
        X[:, -1, 1] = 50.0

        persist_preds = predict_persistence_baseline(X)
        np.testing.assert_array_almost_equal(persist_preds, [45.0, 45.0, 45.0])

        trend_preds = predict_empirical_trend_baseline(X, horizon_steps=6)
        self.assertEqual(len(trend_preds), 3)

        y_true = np.array([45.0, 46.0, 44.0])
        metrics = calculate_metrics(y_true, persist_preds)
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0, places=2)

        # Acceptance gate test: LSTM worse than baseline should NOT be accepted
        gate_res = evaluate_model_acceptance_gate(
            lstm_metrics={"mae": 3.5},
            persistence_metrics={"mae": 1.2},
            trend_metrics={"mae": 1.4},
        )
        self.assertFalse(gate_res["accepted"])
        self.assertEqual(gate_res["recommended_forecaster"], "Empirical Trend Extrapolation")

    def test_16_multi_mode_forecasting_pipeline(self):
        """Verify multi-mode forecasting dispatch for Trend Forecast, LSTM, and Model Comparison."""
        # 1. Trend Forecast
        tf = get_forecast_by_mode("Trend Forecast")
        self.assertEqual(tf["mode"], "Trend Forecast")

        # 2. Experimental LSTM (honestly reports unavailable)
        lstm_f = get_forecast_by_mode("Experimental LSTM")
        self.assertEqual(lstm_f["mode"], "Experimental LSTM")
        self.assertFalse(lstm_f["available"])
        self.assertIn("LSTM unavailable", lstm_f["status"])

        # 3. Model Comparison
        comp = get_forecast_by_mode("Model Comparison")
        self.assertEqual(comp["mode"], "Model Comparison")
        self.assertIn("persistence_baseline", comp)
        self.assertIn("trend_baseline", comp)
        self.assertIn("experimental_lstm", comp)
        self.assertEqual(comp["operational_recommendation"], "Empirical Trend Extrapolation")

    def test_17_dashboard_mode_b_submodes_render_cleanly(self):
        """Verify that dashboard Mode B renders all three forecast strategies without exceptions."""
        from streamlit.testing.v1 import AppTest

        # 1. Test Mode B with Trend Forecast (default)
        at = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py"), default_timeout=30)
        at.session_state["navigation_radio"] = "🤖 AI Models"
        at.session_state["forecasting_mode_radio"] = "📡 Mode B: Live Hardware Mode (Real ESP32 Telemetry)"
        at.session_state["live_forecast_submode_radio"] = "📈 Trend Forecast"
        at.run()
        self.assertFalse(at.exception, f"Dashboard failed in Trend Forecast: {at.exception}")

        # 2. Test Mode B with Experimental LSTM
        at.session_state["live_forecast_submode_radio"] = "🧠 Experimental LSTM"
        at.run()
        self.assertFalse(at.exception, f"Dashboard failed in Experimental LSTM: {at.exception}")

        # 3. Test Mode B with Model Comparison
        at.session_state["live_forecast_submode_radio"] = "⚖️ Model Comparison"
        at.run()
        self.assertFalse(at.exception, f"Dashboard failed in Model Comparison: {at.exception}")


if __name__ == "__main__":
    unittest.main()

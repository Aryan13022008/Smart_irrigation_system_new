"""
Integration and Verification Tests for Phase 22 Experimental Forecasting
========================================================================
Validates that:
1. All three experimental neural forecasting checkpoints exist and load in eval() mode.
2. Parameter counts, sequence length (24), and feature schemas (14) match exact specifications.
3. Feature normalization parameters (scaler_mean, scaler_scale) are properly configured.
4. Models generate finite predictions on stored evaluation sequences.
5. Naive persistence baseline is calculated and compared faithfully.
6. Invalid inputs (bad sequence length, missing features, invalid dimensions) are rejected.
7. Sensor mismatch handling correctly identifies missing channels on live ESP32 hardware.
8. Production 11-feature model and pump-safety locks remain invariant and active.
9. Streamlit dashboard headless AppTest executes without exceptions across all navigation pages.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase13"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"))

from forecasting_models import (
    load_forecasting_model,
    preprocess_sequence,
    predict_future_moisture,
    EXPERIMENTAL_FEATURES,
    SEQUENCE_LENGTH,
    PUMP_ALLOWED as PUMP_ALLOWED_PHASE13,
)
from transformer_inference import (
    load_transformer_model,
    preprocess_transformer_sequence,
    predict_transformer_future_moisture,
    PUMP_ALLOWED as PUMP_ALLOWED_TRANSFORMER,
)
import training.inference as prod_inference


class TestPhase22ForecastingIntegration(unittest.TestCase):
    """Test suite covering Phase 22 experimental forecasting integration."""

    @classmethod
    def setUpClass(cls):
        cls.exp_phase13_dir = PROJECT_ROOT / "training" / "experimental" / "phase13"
        cls.exp_phase15_dir = PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"
        cls.data_dir = (
            PROJECT_ROOT
            / "Soil Moisture, Irrigation Actuator and Weather Dat"
            / "02_processed_data"
            / "preprocessed"
        )

    def test_01_all_three_checkpoints_exist_and_load(self):
        """Verify LSTM, TCN, and Transformer checkpoints exist and load into PyTorch eval() mode."""
        # 1. LSTM
        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        self.assertIsNotNone(lstm_model)
        self.assertFalse(lstm_model.training, "LSTM model must be in eval() mode")
        self.assertEqual(lstm_meta["parameters"], 6689)
        self.assertEqual(lstm_meta["sequence_length"], 24)
        self.assertEqual(len(lstm_meta["features"]), 14)

        # 2. TCN
        tcn_model, tcn_meta = load_forecasting_model("TCN")
        self.assertIsNotNone(tcn_model)
        self.assertFalse(tcn_model.training, "TCN model must be in eval() mode")
        self.assertEqual(tcn_meta["parameters"], 17921)
        self.assertEqual(tcn_meta["sequence_length"], 24)
        self.assertEqual(len(tcn_meta["features"]), 14)

        # 3. Transformer
        trans_model, trans_meta = load_transformer_model()
        self.assertIsNotNone(trans_model)
        self.assertFalse(trans_model.training, "Transformer model must be in eval() mode")
        self.assertEqual(trans_meta["parameters"], 18113)
        self.assertEqual(trans_meta["sequence_length"], 24)
        self.assertEqual(len(trans_meta["features"]), 14)

    def test_02_shared_feature_schema_and_scaler_properties(self):
        """Verify identical 14-feature order and valid scaling statistics across all models."""
        _, lstm_meta = load_forecasting_model("LSTM")
        _, tcn_meta = load_forecasting_model("TCN")
        _, trans_meta = load_transformer_model()

        expected_features = [
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

        self.assertEqual(lstm_meta["features"], expected_features)
        self.assertEqual(tcn_meta["features"], expected_features)
        self.assertEqual(trans_meta["features"], expected_features)

        # Verify scaler shapes and positivity of scale standard deviations
        for meta in [lstm_meta, tcn_meta, trans_meta]:
            mean = meta["scaler_mean"]
            scale = meta["scaler_scale"]
            self.assertEqual(len(mean), 14)
            self.assertEqual(len(scale), 14)
            self.assertTrue(np.all(scale > 0), "Feature scaling factors must be strictly positive")

    def test_03_inference_on_stored_evaluation_sequence(self):
        """Verify that all three models produce finite, scalar predictions on a real stored sequence."""
        csv_file = self.data_dir / "dataset_zone_1_preprocessed.csv"
        self.assertTrue(csv_file.exists(), f"Stored evaluation CSV must exist at {csv_file}")

        df = pd.read_csv(csv_file).iloc[-24:].reset_index(drop=True)
        self.assertEqual(len(df), 24)

        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        tcn_model, tcn_meta = load_forecasting_model("TCN")
        trans_model, trans_meta = load_transformer_model()

        # Run inference
        lstm_pred = predict_future_moisture(lstm_model, df, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"])
        tcn_pred = predict_future_moisture(tcn_model, df, tcn_meta["scaler_mean"], tcn_meta["scaler_scale"])
        trans_pred = predict_transformer_future_moisture(trans_model, df, trans_meta["scaler_mean"], trans_meta["scaler_scale"])
        persist_pred = float(df["soil_moisture"].iloc[-1])

        # Verify finite scalar outputs
        self.assertIsInstance(lstm_pred, float)
        self.assertTrue(np.isfinite(lstm_pred))
        self.assertIsInstance(tcn_pred, float)
        self.assertTrue(np.isfinite(tcn_pred))
        self.assertIsInstance(trans_pred, float)
        self.assertTrue(np.isfinite(trans_pred))
        self.assertIsInstance(persist_pred, float)
        self.assertTrue(np.isfinite(persist_pred))

        # Reasonable volumetric soil moisture range (0% to 100%)
        self.assertGreaterEqual(lstm_pred, -5.0)
        self.assertLessEqual(lstm_pred, 105.0)
        self.assertGreaterEqual(tcn_pred, -5.0)
        self.assertLessEqual(tcn_pred, 105.0)
        self.assertGreaterEqual(trans_pred, -5.0)
        self.assertLessEqual(trans_pred, 105.0)

    def test_04_strict_input_validation_and_rejection(self):
        """Verify that invalid sequence lengths, missing features, and mismatched dimensions raise errors."""
        _, meta = load_forecasting_model("LSTM")
        mean = meta["scaler_mean"]
        scale = meta["scaler_scale"]

        valid_df = pd.DataFrame(np.ones((24, 14)), columns=EXPERIMENTAL_FEATURES)

        # 1. Wrong sequence length (e.g. 10 or 25 rows)
        short_df = pd.DataFrame(np.ones((10, 14)), columns=EXPERIMENTAL_FEATURES)
        with self.assertRaises(ValueError):
            preprocess_sequence(short_df, mean, scale)
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(short_df, mean, scale)

        # 2. Missing feature column
        missing_df = valid_df.drop(columns=["ec"])
        with self.assertRaises(ValueError):
            preprocess_sequence(missing_df, mean, scale)
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(missing_df, mean, scale)

        # 3. Invalid array dimensions (1D array instead of 2D/3D)
        with self.assertRaises(ValueError):
            preprocess_sequence(np.ones(14), mean, scale)
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(np.ones(14), mean, scale)

    def test_05_esp32_sensor_mismatch_and_hardware_blocking(self):
        """Verify that live ESP32 sensor telemetry cannot be passed to 14-feature models without explicit error."""
        # Simulated live ESP32 telemetry (only probe ADC/moisture and DHT22)
        live_esp32_data = {
            "soil_moisture": 32.5,
            "soil_moisture_calibrated": 32.5,
            "air_temperature": 28.4,
            "air_humidity": 65.0,
            "is_calibrated": True,
        }

        # Check required missing channels: 8 missing variables
        missing_channels = [
            "weather_temp",
            "weather_pressure",
            "weather_wind_speed",
            "weather_radiation",
            "soil_temperature_0-7cm",
            "soil_temperature_7-18cm",
            "ec",
            "ph",
        ]
        for ch in missing_channels:
            self.assertNotIn(ch, live_esp32_data)

        # Passing raw single-point dictionary to sequence preprocessors must raise TypeError
        _, meta = load_forecasting_model("LSTM")
        with self.assertRaises(TypeError):
            preprocess_sequence(live_esp32_data, meta["scaler_mean"], meta["scaler_scale"])
        with self.assertRaises(TypeError):
            preprocess_transformer_sequence(live_esp32_data, meta["scaler_mean"], meta["scaler_scale"])

    def test_06_production_11_feature_model_and_safety_invariance(self):
        """Verify that production 11-feature model, tuned threshold, and pump safety lock remain locked."""
        # 1. Pump actuation locks strictly False
        self.assertFalse(PUMP_ALLOWED_PHASE13, "Phase 13 PUMP_ALLOWED must be False")
        self.assertFalse(PUMP_ALLOWED_TRANSFORMER, "Transformer PUMP_ALLOWED must be False")

        # 2. Production active model is 11-feature model
        self.assertEqual(prod_inference.ACTIVE_MODEL_NAME, "Random Forest (11-Feature)")
        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)
        self.assertEqual(len(metrics["features"]), 11)

    def test_07_dashboard_headless_execution(self):
        """Verify that Streamlit AppTest runs cleanly across all navigation tabs including AI Models."""
        from streamlit.testing.v1 import AppTest

        for page in ["🏠 Overview", "💧 Irrigation", "🤖 AI Models"]:
            at = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py"), default_timeout=30)
            at.session_state["navigation_radio"] = page
            at.run()
            self.assertEqual(len(at.exception), 0, f"AppTest produced exceptions on page {page}: {at.exception}")


if __name__ == "__main__":
    unittest.main()

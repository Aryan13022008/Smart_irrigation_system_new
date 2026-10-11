"""
Unit and Integration Tests for Removing Forecasting Blocker & Enabling Models
==============================================================================
Validates that:
1. All three experimental checkpoints (LSTM, TCN, Transformer) exist and load in eval() mode
   with exact parameters (6,689, 17,921, 18,113).
2. Mode A (Offline Research Mode) executes real PyTorch inference on stored evaluation sequences
   across Sectors 1, 2, 4, and 5, generating finite scalar continuous forecasts and persistence baseline.
3. Mode B (Live Hardware Mode) safely verifies hardware sequence availability and strictly rejects
   incompatible schemas (missing 8 channels) with informative ValueError without fabricating data.
4. Decoupling verification: Missing live hardware features in Mode B do not block or crash Mode A execution.
5. Safety invariants: Production 11-feature model, tuned threshold (0.3270985223048995), and
   PUMP_ALLOWED=False remain strictly preserved and invariant.
6. Streamlit headless AppTest executes cleanly on "🤖 AI Models" tab in both Mode A and Mode B.
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
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase25"))

from forecasting_models import (
    load_forecasting_model,
    predict_future_moisture,
    preprocess_sequence,
    EXPERIMENTAL_FEATURES,
    SEQUENCE_LENGTH,
    PUMP_ALLOWED as PUMP_ALLOWED_PHASE13,
)
from transformer_inference import (
    load_transformer_model,
    predict_transformer_future_moisture,
    preprocess_transformer_sequence,
    PUMP_ALLOWED as PUMP_ALLOWED_TRANSFORMER,
)
from realtime_history import extract_realtime_24step_hardware_sequence
import training.inference as prod_inference


class TestRemoveForecastingBlock(unittest.TestCase):
    """Test suite confirming forecasting blockers are removed and models are runnable."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = (
            PROJECT_ROOT
            / "Soil Moisture, Irrigation Actuator and Weather Dat"
            / "02_processed_data"
            / "preprocessed"
        )
        cls.checkpoints = {
            "LSTM": PROJECT_ROOT / "training" / "experimental" / "phase13" / "lstm_experimental.pt",
            "TCN": PROJECT_ROOT / "training" / "experimental" / "phase13" / "tcn_experimental.pt",
            "Transformer": (
                PROJECT_ROOT
                / "training"
                / "experimental"
                / "phase15_transformer"
                / "transformer_experimental.pt"
            ),
        }

    def test_01_checkpoints_exist_and_load_eval_mode(self):
        """All three checkpoints must exist, be non-empty, and load into eval() mode."""
        for name, path in self.checkpoints.items():
            self.assertTrue(path.exists(), f"{name} checkpoint file missing: {path}")
            self.assertGreater(path.stat().st_size, 0, f"{name} checkpoint file is empty")

        # LSTM
        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        self.assertFalse(lstm_model.training, "LSTM model must be in eval() mode")
        self.assertEqual(lstm_meta["parameters"], 6689)
        self.assertEqual(lstm_meta["sequence_length"], 24)

        # TCN
        tcn_model, tcn_meta = load_forecasting_model("TCN")
        self.assertFalse(tcn_model.training, "TCN model must be in eval() mode")
        self.assertEqual(tcn_meta["parameters"], 17921)
        self.assertEqual(tcn_meta["sequence_length"], 24)

        # Transformer
        trans_model, trans_meta = load_transformer_model()
        self.assertFalse(trans_model.training, "Transformer model must be in eval() mode")
        self.assertEqual(trans_meta["parameters"], 18113)
        self.assertEqual(trans_meta["sequence_length"], 24)

    def test_02_mode_a_stored_sequences_inference(self):
        """Mode A runs real PyTorch inference on genuine 24-step sequences across all stored sectors."""
        sectors = [1, 2, 4, 5]
        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        tcn_model, tcn_meta = load_forecasting_model("TCN")
        trans_model, trans_meta = load_transformer_model()

        for sector in sectors:
            csv_path = self.data_dir / f"dataset_zone_{sector}_preprocessed.csv"
            self.assertTrue(csv_path.exists(), f"Stored dataset for sector {sector} must exist")

            df = pd.read_csv(csv_path).iloc[-24:].reset_index(drop=True)
            self.assertEqual(len(df), SEQUENCE_LENGTH)
            for feat in EXPERIMENTAL_FEATURES:
                self.assertIn(feat, df.columns)

            # Execute real inference
            p_lstm = predict_future_moisture(
                lstm_model, df, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"]
            )
            p_tcn = predict_future_moisture(
                tcn_model, df, tcn_meta["scaler_mean"], tcn_meta["scaler_scale"]
            )
            p_trans = predict_transformer_future_moisture(
                trans_model, df, trans_meta["scaler_mean"], trans_meta["scaler_scale"]
            )
            p_persist = float(df["soil_moisture"].iloc[-1])

            # Validate outputs are finite numbers
            for name, val in [("LSTM", p_lstm), ("TCN", p_tcn), ("Transformer", p_trans), ("Persistence", p_persist)]:
                self.assertIsInstance(val, float, f"{name} forecast must be float")
                self.assertTrue(np.isfinite(val), f"{name} forecast must be finite")
                self.assertGreaterEqual(val, -5.0)
                self.assertLessEqual(val, 105.0)

    def test_03_mode_b_rejection_of_incompatible_hardware_sequence(self):
        """Mode B checks live observation buffer and rejects sequence lacking 8 channels with ValueError."""
        hw_seq = extract_realtime_24step_hardware_sequence()
        self.assertIn("available", hw_seq)
        self.assertIn("streak_length", hw_seq)

        # If 24-step sequence is extracted from SQLite, test that feeding it to models fails with ValueError
        if hw_seq["available"] and hw_seq["sequence_df"] is not None:
            df_hw = hw_seq["sequence_df"]
            self.assertEqual(len(df_hw), 24)

            _, lstm_meta = load_forecasting_model("LSTM")
            with self.assertRaises(ValueError) as ctx:
                preprocess_sequence(df_hw, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"])
            self.assertIn("missing required features", str(ctx.exception).lower())

            with self.assertRaises(ValueError) as ctx_trans:
                preprocess_transformer_sequence(df_hw, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"])
            self.assertIn("missing required features", str(ctx_trans.exception).lower())
        else:
            # Buffer accumulating blocker is clearly reported
            self.assertIsNotNone(hw_seq["blocker"])

    def test_04_decoupling_mode_a_from_mode_b_hardware_state(self):
        """Mode A executes reliably even when Mode B live hardware data has missing channels or accumulating buffer."""
        # Hardware buffer may be incomplete or missing 8 channels
        hw_seq = extract_realtime_24step_hardware_sequence()

        # Mode A must execute without error regardless of hw_seq state
        csv_path = self.data_dir / "dataset_zone_1_preprocessed.csv"
        df = pd.read_csv(csv_path).iloc[-24:].reset_index(drop=True)
        lstm_model, lstm_meta = load_forecasting_model("LSTM")

        pred = predict_future_moisture(
            lstm_model, df, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"]
        )
        self.assertTrue(np.isfinite(pred))

    def test_05_safety_invariants_preserved(self):
        """Safety rules strictly verified: PUMP_ALLOWED is False, 11-feature model active with tuned threshold."""
        self.assertFalse(PUMP_ALLOWED_PHASE13)
        self.assertFalse(PUMP_ALLOWED_TRANSFORMER)

        self.assertEqual(prod_inference.ACTIVE_MODEL_NAME, "Random Forest (11-Feature)")
        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)
        self.assertEqual(len(metrics["features"]), 11)

    def test_06_dashboard_apptest_both_modes(self):
        """Streamlit AppTest executes cleanly on AI Models page with both Mode A and Mode B selected."""
        from streamlit.testing.v1 import AppTest

        # Test Mode A execution
        at_a = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py"), default_timeout=30)
        at_a.session_state["navigation_radio"] = "🤖 AI Models"
        at_a.session_state["forecasting_mode_radio"] = "🔬 Mode A: Offline Research Mode (Compatible Stored Dataset)"
        at_a.run()
        self.assertEqual(len(at_a.exception), 0, f"AppTest threw exception in Mode A: {at_a.exception}")

        # Test Mode B execution
        at_b = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py"), default_timeout=30)
        at_b.session_state["navigation_radio"] = "🤖 AI Models"
        at_b.session_state["forecasting_mode_radio"] = "📡 Mode B: Live Hardware Mode (Real ESP32 Telemetry)"
        at_b.run()
        self.assertEqual(len(at_b.exception), 0, f"AppTest threw exception in Mode B: {at_b.exception}")


if __name__ == "__main__":
    unittest.main()

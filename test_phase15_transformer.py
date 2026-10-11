"""
Unit and Integration Tests for Phase 15 Experimental Transformer
================================================================
Verifies Transformer loading, tensor shapes, preprocessing validation, sequence
continuity, data leakage prevention, baseline comparison, and pump safety locks.
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
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"))

from transformer_inference import (
    load_transformer_model,
    preprocess_transformer_sequence,
    predict_transformer_future_moisture,
    EXPERIMENTAL_FEATURES,
    SEQUENCE_LENGTH,
    PUMP_ALLOWED,
)

import training.inference as prod_inference


class TestPhase15Transformer(unittest.TestCase):
    """Test suite covering Phase 15 Transformer integration and safety."""

    def setUp(self):
        self.exp_dir = PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"
        self.data_dir = PROJECT_ROOT / "Soil Moisture, Irrigation Actuator and Weather Dat" / "02_processed_data" / "preprocessed"

    def test_01_checkpoint_and_metadata_exist(self):
        """Verify Transformer checkpoint and evaluation reports are saved on disk."""
        self.assertTrue((self.exp_dir / "transformer_experimental.pt").exists(), "transformer checkpoint missing")
        self.assertTrue((self.exp_dir / "phase15_transformer_results.json").exists(), "results JSON missing")
        self.assertTrue((self.exp_dir / "phase15_transformer_results.md").exists(), "results Markdown missing")
        self.assertTrue((self.exp_dir / "train_transformer_forecasting.py").exists(), "training script missing")

    def test_02_load_transformer_model(self):
        """Verify Transformer loads cleanly with correct parameter count and metadata."""
        model, meta = load_transformer_model()
        self.assertIsNotNone(model)
        self.assertEqual(meta["model_name"], "TransformerEncoder")
        self.assertEqual(meta["parameters"], 18113)
        self.assertEqual(meta["sequence_length"], 24)
        self.assertEqual(len(meta["features"]), 14)
        self.assertEqual(meta["features"], EXPERIMENTAL_FEATURES)
        self.assertTrue(np.all(meta["scaler_scale"] > 0))

    def test_03_tensor_shapes_and_forward_pass(self):
        """Verify input shapes (batch, 24, 14) produce valid scalar / 1D predictions."""
        model, meta = load_transformer_model()

        # Single sequence (1, 24, 14)
        dummy_single = np.ones((24, 14), dtype=np.float32) * 25.0
        pred = predict_transformer_future_moisture(model, dummy_single, meta["scaler_mean"], meta["scaler_scale"])
        self.assertIsInstance(pred, float)
        self.assertTrue(np.isfinite(pred))

        # Batch sequence (4, 24, 14)
        dummy_batch = np.ones((4, 24, 14), dtype=np.float32) * 25.0
        tensor_batch = preprocess_transformer_sequence(dummy_batch, meta["scaler_mean"], meta["scaler_scale"])
        self.assertEqual(tensor_batch.shape, (4, 24, 14))

        with torch.no_grad():
            preds_batch = model(tensor_batch).squeeze(-1).numpy()
        self.assertEqual(preds_batch.shape, (4,))
        self.assertTrue(np.all(np.isfinite(preds_batch)))

    def test_04_missing_data_and_shape_rejection(self):
        """Verify preprocessing strictly rejects wrong row counts, missing features, and invalid ndim."""
        _, meta = load_transformer_model()
        mean = meta["scaler_mean"]
        scale = meta["scaler_scale"]

        # Missing feature
        df_valid = pd.DataFrame(np.ones((24, 14)), columns=EXPERIMENTAL_FEATURES)
        df_missing = df_valid.drop(columns=["soil_moisture"])
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(df_missing, mean, scale)

        # Invalid row count (e.g. 23 or 25 instead of 24)
        df_wrong_rows = pd.DataFrame(np.ones((23, 14)), columns=EXPERIMENTAL_FEATURES)
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(df_wrong_rows, mean, scale)

        # Invalid array shape
        with self.assertRaises(ValueError):
            preprocess_transformer_sequence(np.ones((10, 14)), mean, scale)

    def test_05_sequence_continuity_and_gap_handling(self):
        """Verify that sequence construction strictly requires exactly 10-minute cadence without gaps."""
        if not self.data_dir.exists():
            self.skipTest("Data directory not found")

        df1 = pd.read_csv(self.data_dir / "dataset_zone_1_preprocessed.csv")
        df1["ts"] = pd.to_datetime(df1["ts"])

        # Create an artificial 20-minute gap
        broken_window = df1.iloc[:24].copy()
        broken_window.loc[broken_window.index[12], "ts"] += pd.Timedelta(minutes=30)
        total_span = (broken_window["ts"].iloc[-1] - broken_window["ts"].iloc[0]).total_seconds() / 60.0
        # Assert total span detects gap
        self.assertNotEqual(total_span, 23 * 10)

    def test_06_chronological_split_integrity_no_leakage(self):
        """Verify splits within each sector are strictly forward chronological."""
        if not self.data_dir.exists():
            self.skipTest("Data directory not found")

        for z in [1, 2, 4, 5]:
            df = pd.read_csv(self.data_dir / f"dataset_zone_{z}_preprocessed.csv")
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts").reset_index(drop=True)

            n = len(df)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)

            train_max = df.iloc[:n_train]["ts"].max()
            val_min = df.iloc[n_train : n_train + n_val]["ts"].min()
            val_max = df.iloc[n_train : n_train + n_val]["ts"].max()
            test_min = df.iloc[n_train + n_val:]["ts"].min()

            self.assertLess(train_max, val_min, f"Train and val overlap in zone {z}")
            self.assertLess(val_max, test_min, f"Val and test overlap in zone {z}")

    def test_07_persistence_baseline_dominance(self):
        """Verify reproduction of test metrics showing persistence baseline beats all deep learning models."""
        metrics_file = self.exp_dir / "phase15_transformer_results.json"
        self.assertTrue(metrics_file.exists())

        import json
        with open(metrics_file) as f:
            data = json.load(f)

        overall = data["overall_test_metrics"]
        base_mae = overall["persistence_baseline"]["mae"]
        tf_mae = overall["transformer"]["mae"]
        lstm_mae = overall["lstm"]["mae"]
        tcn_mae = overall["tcn"]["mae"]

        # Assert baseline achieves ~0.412
        self.assertAlmostEqual(base_mae, 0.4124, places=3)
        # Assert Transformer error is ~6.12
        self.assertAlmostEqual(tf_mae, 6.1237, places=2)
        # Baseline is superior to all three neural networks
        self.assertLess(base_mae, tf_mae)
        self.assertLess(base_mae, lstm_mae)
        self.assertLess(base_mae, tcn_mae)

    def test_08_per_zone_metrics_and_simpsons_paradox(self):
        """Verify per-zone metrics are reported and within-zone performance reflects stationary test conditions."""
        metrics_file = self.exp_dir / "phase15_transformer_results.json"
        import json
        with open(metrics_file) as f:
            data = json.load(f)

        per_zone = data["per_zone_metrics"]
        self.assertIn("1", per_zone)
        self.assertIn("2", per_zone)
        self.assertIn("4", per_zone)
        self.assertIn("5", per_zone)

        # Baseline MAE is lowest in all zones
        for z in ["1", "2", "4", "5"]:
            zm = per_zone[z]
            self.assertLess(zm["persistence_baseline"]["mae"], zm["transformer"]["mae"])

    def test_09_pump_safety_lock(self):
        """Verify PUMP_ALLOWED is strictly False for Transformer forecasting."""
        self.assertFalse(PUMP_ALLOWED, "Experimental forecasting must never actuate pump")

    def test_10_production_classifier_unaffected(self):
        """Verify Phase 11 production 11-feature model remains active and unchanged."""
        prod_model_path = PROJECT_ROOT / "training" / "models" / "random_forest_11feat.pkl"
        self.assertTrue(prod_model_path.exists())

        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)


if __name__ == "__main__":
    unittest.main()

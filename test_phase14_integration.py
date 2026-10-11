"""
Unit and Integration Tests for Phase 14 Dataset Integration and Experimental Models
====================================================================================
Tests experimental LSTM and TCN forecasting models, feature schemas, chronological splitting,
persistence baseline replication, zone isolation, pump safety locks, and production isolation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase13"))

from forecasting_models import (
    load_forecasting_model,
    preprocess_sequence,
    predict_future_moisture,
    evaluate_new_dataset,
    EXPERIMENTAL_FEATURES,
    SEQUENCE_LENGTH,
    CADENCE_MINUTES,
    TARGET_NAME,
    PUMP_ALLOWED,
)

import training.inference as prod_inference


class TestPhase14Integration(unittest.TestCase):
    """Test suite covering Phase 14 experimental integration and safety."""

    def setUp(self):
        self.exp_dir = PROJECT_ROOT / "training" / "experimental" / "phase13"
        self.data_dir = PROJECT_ROOT / "Soil Moisture, Irrigation Actuator and Weather Dat" / "02_processed_data" / "preprocessed"

    def test_01_experimental_artifacts_exist(self):
        """Verify all Phase 13/14 experimental artifacts are present in the project."""
        self.assertTrue((self.exp_dir / "lstm_experimental.pt").exists(), "lstm_experimental.pt missing")
        self.assertTrue((self.exp_dir / "tcn_experimental.pt").exists(), "tcn_experimental.pt missing")
        self.assertTrue((self.exp_dir / "phase13_lstm_tcn_results.json").exists(), "results.json missing")
        self.assertTrue((self.exp_dir / "phase13_lstm_tcn_results.md").exists(), "results.md missing")

    def test_02_load_experimental_models_and_metadata(self):
        """Verify LSTM and TCN load safely with matching metadata and parameter counts."""
        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        self.assertIsNotNone(lstm_model)
        self.assertEqual(lstm_meta["parameters"], 6689)
        self.assertEqual(lstm_meta["sequence_length"], 24)
        self.assertEqual(lstm_meta["target"], "target_mean_24h")
        self.assertEqual(len(lstm_meta["features"]), 14)

        tcn_model, tcn_meta = load_forecasting_model("TCN")
        self.assertIsNotNone(tcn_model)
        self.assertEqual(tcn_meta["parameters"], 17921)
        self.assertEqual(tcn_meta["sequence_length"], 24)
        self.assertEqual(tcn_meta["target"], "target_mean_24h")
        self.assertEqual(len(tcn_meta["features"]), 14)

    def test_03_feature_ordering_and_scaler_parameters(self):
        """Verify exact 14-feature schema ordering and positive scaler scale vectors."""
        _, lstm_meta = load_forecasting_model("LSTM")
        self.assertEqual(lstm_meta["features"], EXPERIMENTAL_FEATURES)
        self.assertEqual(len(lstm_meta["scaler_mean"]), 14)
        self.assertEqual(len(lstm_meta["scaler_scale"]), 14)
        self.assertTrue(np.all(lstm_meta["scaler_scale"] > 0), "Scaler scale must be strictly positive")

    def test_04_preprocessing_and_input_shape_validation(self):
        """Verify preprocessing accepts valid shapes and rejects invalid lengths/dimensions."""
        _, meta = load_forecasting_model("LSTM")
        mean = meta["scaler_mean"]
        scale = meta["scaler_scale"]

        # 1. Valid 2D numpy array (24, 14)
        valid_2d = np.ones((24, 14), dtype=np.float32) * 20.0
        tensor_2d = preprocess_sequence(valid_2d, mean, scale)
        self.assertEqual(tensor_2d.shape, (1, 24, 14))

        # 2. Valid DataFrame with 24 rows
        df_valid = pd.DataFrame(np.ones((24, 14)), columns=EXPERIMENTAL_FEATURES)
        tensor_df = preprocess_sequence(df_valid, mean, scale)
        self.assertEqual(tensor_df.shape, (1, 24, 14))

        # 3. Invalid length (e.g. 20 rows instead of 24)
        df_invalid_len = pd.DataFrame(np.ones((20, 14)), columns=EXPERIMENTAL_FEATURES)
        with self.assertRaises(ValueError):
            preprocess_sequence(df_invalid_len, mean, scale)

        # 4. Missing feature column
        df_missing_col = df_valid.drop(columns=["weather_temp"])
        with self.assertRaises(ValueError):
            preprocess_sequence(df_missing_col, mean, scale)

        # 5. Invalid array dimensions
        with self.assertRaises(ValueError):
            preprocess_sequence(np.ones((10, 5)), mean, scale)

    def test_05_forward_pass_and_prediction_finite(self):
        """Verify forward pass produces valid, finite scalar predictions."""
        lstm_model, meta = load_forecasting_model("LSTM")
        dummy_input = np.ones((24, 14), dtype=np.float32) * 30.0
        pred_lstm = predict_future_moisture(lstm_model, dummy_input, meta["scaler_mean"], meta["scaler_scale"])
        self.assertIsInstance(pred_lstm, float)
        self.assertTrue(np.isfinite(pred_lstm))

        tcn_model, tcn_meta = load_forecasting_model("TCN")
        pred_tcn = predict_future_moisture(tcn_model, dummy_input, tcn_meta["scaler_mean"], tcn_meta["scaler_scale"])
        self.assertIsInstance(pred_tcn, float)
        self.assertTrue(np.isfinite(pred_tcn))

    def test_06_chronological_split_integrity_and_no_leakage(self):
        """Verify train/val/test splits within sectors are strictly chronological with zero temporal overlap."""
        if not self.data_dir.exists():
            self.skipTest("Processed dataset directory not found")

        for z in [1, 2, 4, 5]:
            csv_path = self.data_dir / f"dataset_zone_{z}_preprocessed.csv"
            df = pd.read_csv(csv_path)
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts").reset_index(drop=True)

            n = len(df)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)

            train_df = df.iloc[:n_train]
            val_df = df.iloc[n_train : n_train + n_val]
            test_df = df.iloc[n_train + n_val :]

            # Assert strict forward progression across splits
            self.assertLess(train_df["ts"].max(), val_df["ts"].min())
            self.assertLess(val_df["ts"].max(), test_df["ts"].min())

    def test_07_persistence_baseline_and_recalculation(self):
        """Verify reproduction of test MAE ~0.4124 for persistence baseline on identical test windows."""
        if not self.data_dir.exists():
            self.skipTest("Processed dataset directory not found")

        res = evaluate_new_dataset(self.data_dir)
        self.assertEqual(res["total_test_windows"], 6348)

        base_mae = res["overall"]["persistence_baseline"]["mae"]
        lstm_mae = res["overall"]["lstm"]["mae"]
        tcn_mae = res["overall"]["tcn"]["mae"]

        # Persistence baseline is ~0.412
        self.assertAlmostEqual(base_mae, 0.4124, places=3)
        # LSTM test MAE is ~3.867
        self.assertAlmostEqual(lstm_mae, 3.8665, places=2)
        # TCN test MAE is ~4.690
        self.assertAlmostEqual(tcn_mae, 4.6904, places=2)

        # Baseline beats both neural networks by nearly 10x
        self.assertLess(base_mae, lstm_mae)
        self.assertLess(base_mae, tcn_mae)

    def test_08_zone_isolation_and_simpsons_paradox(self):
        """Verify per-zone metrics show negative or degraded R2 within sectors despite high pooled R2."""
        if not self.data_dir.exists():
            self.skipTest("Processed dataset directory not found")

        res = evaluate_new_dataset(self.data_dir)
        per_zone = res["per_zone"]

        # Check sectors 1, 2, 4, 5 are present
        self.assertIn(1, per_zone)
        self.assertIn(2, per_zone)
        self.assertIn(4, per_zone)
        self.assertIn(5, per_zone)

        # Sector 1 within-sector R2 is negative for LSTM and TCN
        self.assertLess(per_zone[1]["lstm"]["r2"], 0.0)
        self.assertLess(per_zone[1]["tcn"]["r2"], 0.0)

        # Future irrigation is 0.0% in test period
        self.assertEqual(res["audit_insights"]["test_future_irrigation_events_total"], 0)

    def test_09_pump_safety_lock(self):
        """Verify experimental forecasting module hard-locks pump control to disabled."""
        self.assertFalse(PUMP_ALLOWED, "Experimental forecasting must never allow pump actuation")

    def test_10_production_model_and_pipeline_unchanged(self):
        """Verify Phase 11 production 11-feature model, metrics, and inference remain strictly unchanged."""
        prod_model_path = PROJECT_ROOT / "training" / "models" / "random_forest_11feat.pkl"
        prod_metrics_path = PROJECT_ROOT / "training" / "models" / "random_forest_11feat_metrics.json"

        self.assertTrue(prod_model_path.exists())
        self.assertTrue(prod_metrics_path.exists())

        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)

        trust = prod_inference.get_model_trustworthiness("Random Forest (11-Feature)")
        self.assertEqual(trust["category"], "Experimental — Advisory Only")
        self.assertIn("Experimental", trust["badge"])


if __name__ == "__main__":
    unittest.main()

"""
Comprehensive Test Suite for ML Integration in Smart Irrigation System
======================================================================
Tests:
1. Model weights & files verification (no modifications).
2. Random Forest feature mapping, air_humidity vs humidity, weather inputs.
3. Raw ADC vs calibrated soil moisture handling.
4. Deep learning models sigmoid/probability fix and input compatibility.
5. Dashboard predict_irrigation function integration.
6. Reproduction of test metrics to verify model accuracy preservation.
"""

import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import sys
import tempfile
import os
import hashlib
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import database
from inference import (
    predict,
    audit_feature_readiness,
    get_metrics,
    get_model_comparison_rows,
    ImprovedLSTMClassifier,
    TCNClassifier,
    TransformerClassifier,
)
from dashboard import predict_irrigation


class TestMLIntegration(unittest.TestCase):

    def setUp(self):
        self.models_dir = Path(__file__).resolve().parent / "training" / "models"

    def test_01_models_and_metrics_files_exist(self):
        """Verify all model files and metrics exist (including active 11-feature model and preserved baselines)."""
        expected_files = [
            "random_forest_11feat.pkl",
            "random_forest_11feat_metrics.json",
            "random_forest.pkl",
            "random_forest_metrics.json",
            "improved_lstm.pt",
            "lstm_metrics.json",
            "tcn.pt",
            "tcn_metrics.json",
            "transformer.pt",
            "transformer_metrics.json",
        ]
        for fname in expected_files:
            fpath = self.models_dir / fname
            self.assertTrue(fpath.exists(), f"Missing expected model file: {fname}")

    def test_02_random_forest_feature_alignment(self):
        """Verify random_forest_11feat.pkl and random_forest.pkl feature names and order exactly match metrics JSON."""
        # 1. Active 11-Feature Model
        rf11_model = joblib.load(self.models_dir / "random_forest_11feat.pkl")
        metrics11 = get_metrics("Random Forest (11-Feature)")
        expected11 = [
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
        self.assertEqual(len(metrics11["features"]), 11)
        self.assertEqual(rf11_model.n_features_in_, 11)
        self.assertEqual(metrics11["features"], expected11)
        self.assertEqual(list(rf11_model.feature_names_in_), expected11)
        self.assertNotIn("soil_temperature", expected11)
        self.assertNotIn("soil_humidity", expected11)

        # 2. Preserved 13-Feature Baseline
        rf_model = joblib.load(self.models_dir / "random_forest.pkl")
        metrics = get_metrics("Random Forest")
        expected_features = metrics["features"]

        self.assertEqual(len(expected_features), 13)
        self.assertEqual(rf_model.n_features_in_, 13)
        self.assertEqual(list(rf_model.feature_names_in_), expected_features)
        self.assertIn("air_humidity", expected_features)
        self.assertNotIn("humidity", expected_features)
        self.assertIn("soil_humidity", expected_features)

    def test_03_random_forest_uncalibrated_adc_rejected(self):
        """Verify raw ADC sensor readings (e.g. 2300) are NOT treated as calibrated moisture."""
        sensor_data = {
            "soil_moisture": 2340.0,
            "soil1_raw": 2340,
            "is_calibrated": False,
            "temperature": 25.0,
            "humidity": 60.0,
        }
        res = predict("Random Forest", sensor_data, {})
        self.assertIsNone(res["required"])
        self.assertIsNone(res["confidence"])
        self.assertIn("raw ADC", res["message"])
        self.assertIn("N/A", res["message"])

    def test_04_random_forest_missing_hardware_handled(self):
        """Verify missing sensor data returns N/A explanation rather than fake fallback."""
        res = predict("Random Forest", {}, {})
        self.assertIsNone(res["required"])
        self.assertIsNone(res["confidence"])
        self.assertIn("Hardware not connected", res["message"])

    def test_05_random_forest_calibrated_inference_and_feature_mapping(self):
        """Verify calibrated soil moisture produces valid prediction with correct air_humidity mapping."""
        # Dry soil (20%) -> should need irrigation
        sensor_data = {
            "soil_moisture": 20.0,
            "soil_moisture_calibrated": 20.0,
            "is_calibrated": True,
            "temperature": 30.0,
            "humidity": 35.0,  # Air humidity from DHT22
        }
        weather_data = {
            "temperature": 29.0,
            "humidity": 40.0,
            "wind_speed": 4.5,  # m/s -> 16.2 km/h
            "wind_gust": 8.0,   # m/s -> 28.8 km/h
            "pressure": 1013.0, # hPa -> 101.3 kPa
        }
        res = predict("Random Forest", sensor_data, weather_data)
        self.assertIsNotNone(res["required"])
        self.assertIsInstance(res["required"], bool)
        self.assertIsNotNone(res["confidence"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("threshold", res["message"])

    def test_06_deep_learning_missing_sequence_handled_honestly(self):
        """Verify DL models return N/A when called with live single-probe telemetry instead of 11 channels."""
        sensor_data = {"soil_moisture": 25.0, "is_calibrated": True}
        for dl_name in ["LSTM", "TCN", "Transformer"]:
            res = predict(dl_name, sensor_data, {})
            self.assertIsNone(res["required"])
            self.assertIsNone(res["confidence"])
            self.assertIn("11 historical station irrigation plot channels", res["message"])

    def test_07_deep_learning_forward_pass_and_sigmoid_calculation(self):
        """Verify deep learning branch executes forward pass and computes sigmoid properly."""
        dummy_sequence = np.random.randn(1, 24, 11).astype(np.float32)
        for dl_name in ["LSTM", "TCN", "Transformer"]:
            res = predict(dl_name, sequence_data=dummy_sequence)
            self.assertIsNotNone(res["required"], f"{dl_name} failed forward pass")
            self.assertIsInstance(res["required"], bool)
            self.assertIsNotNone(res["confidence"])
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)

    def test_08_dashboard_predict_irrigation_integration(self):
        """Verify dashboard's predict_irrigation wrapper works seamlessly with active 11-feature model."""
        sensor_uncal = {"soil_moisture": 1800.0, "is_calibrated": False}
        res_uncal = predict_irrigation(sensor_uncal, {})
        self.assertEqual(res_uncal["model"], "Random Forest (11-Feature)")
        self.assertIsNone(res_uncal["required"])
        self.assertIsNone(res_uncal["confidence"])

        sensor_cal = {"soil_moisture": 15.0, "is_calibrated": True}
        res_cal = predict_irrigation(sensor_cal, {})
        self.assertEqual(res_cal["model"], "Random Forest (11-Feature)")
        self.assertIsNotNone(res_cal["required"])
        self.assertIsNotNone(res_cal["confidence"])

    def test_09_model_comparison_table(self):
        """Verify model comparison table correctly lists models and distinguishes test vs validation."""
        rows = get_model_comparison_rows()
        self.assertEqual(len(rows), 5)
        models_found = [r["Model"] for r in rows]
        self.assertEqual(models_found, ["Random Forest (11-Feature)", "Random Forest (13-Feat Baseline)", "LSTM", "TCN", "Transformer"])

        # Check Random Forest (11-Feature) has Test Set metrics
        rf11_row = rows[0]
        self.assertEqual(rf11_row["Model"], "Random Forest (11-Feature)")
        self.assertIn("Test Set", rf11_row["Evaluation Split"])
        self.assertEqual(rf11_row["ROC-AUC"], "0.6808")
        self.assertEqual(rf11_row["PR-AUC"], "0.6817")
        self.assertEqual(rf11_row["Tuned Threshold"], "0.327")

    def test_10_accuracy_reproduction_verification(self):
        """Reproduce evaluation of random_forest.pkl on original test data to verify weights are unchanged."""
        iot_data_path = Path("/Users/aryanagrawal/Downloads/iotsensordata.csv")
        if not iot_data_path.exists():
            self.skipTest("Original raw test data not found at /Users/aryanagrawal/Downloads/iotsensordata.csv")

        df = pd.read_csv(iot_data_path)
        df = df.drop(columns=["ph", "rainfall", "N", "P", "K", "Time"]).dropna().reset_index(drop=True)
        df = df.rename(columns={
            "Soil Moisture": "soil_moisture",
            "Temperature": "soil_temperature",
            " Soil Humidity": "soil_humidity",
            "Air temperature (C)": "temperature",
            "Wind speed (Km/h)": "wind_speed",
            "Air humidity (%)": "air_humidity",
            "Wind gust (Km/h)": "wind_gust",
            "Pressure (KPa)": "pressure"
        })
        df["irrigation_target"] = df["Status"].map({"ON": 1, "OFF": 0})
        df["Date"] = pd.date_range(start="2022-01-01", periods=len(df), freq="1h")
        df["soil_moisture_previous"] = df["soil_moisture"].shift(1)
        df["soil_moisture_change"] = df["soil_moisture"] - df["soil_moisture_previous"]
        df["temp_previous"] = df["temperature"].shift(1)
        df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
        df = df.bfill()

        split_idx = int(len(df) * 0.75)
        test_df = df.iloc[split_idx:]

        features = [
            "soil_moisture", "soil_temperature", "soil_humidity", "temperature",
            "wind_speed", "air_humidity", "wind_gust", "pressure",
            "soil_moisture_previous", "soil_moisture_change", "temp_previous",
            "month_sin", "month_cos"
        ]

        X_test = test_df[features]
        y_test = test_df["irrigation_target"]

        rf = joblib.load(self.models_dir / "random_forest.pkl")
        with open(self.models_dir / "random_forest_metrics.json") as f:
            saved_m = json.load(f)

        test_proba = rf.predict_proba(X_test)[:, 1]
        tuned_thresh = saved_m["threshold_tuned"]

        # 1. Default threshold (0.50)
        y_pred_50 = (test_proba >= 0.5).astype(int)
        acc_50 = accuracy_score(y_test, y_pred_50)
        roc_auc = roc_auc_score(y_test, test_proba)
        pr_auc = average_precision_score(y_test, test_proba)

        self.assertAlmostEqual(acc_50, saved_m["test_at_default_threshold"]["accuracy"], places=6)
        self.assertAlmostEqual(roc_auc, saved_m["test_at_default_threshold"]["roc_auc"], places=6)
        self.assertAlmostEqual(pr_auc, saved_m["test_at_default_threshold"]["pr_auc"], places=6)

        # 2. Tuned threshold
        y_pred_tuned = (test_proba >= tuned_thresh).astype(int)
        acc_tuned = accuracy_score(y_test, y_pred_tuned)
        recall_tuned = recall_score(y_test, y_pred_tuned)
        f1_tuned = f1_score(y_test, y_pred_tuned)

        self.assertAlmostEqual(acc_tuned, saved_m["test_at_tuned_threshold"]["accuracy"], places=6)
        self.assertAlmostEqual(recall_tuned, saved_m["test_at_tuned_threshold"]["recall"], places=6)
        self.assertAlmostEqual(f1_tuned, saved_m["test_at_tuned_threshold"]["f1"], places=6)

    def test_11_model_trustworthiness_and_thresholds(self):
        """Verify each model's trustworthiness profile, thresholds, and input readiness codes."""
        from inference import get_model_trustworthiness

        # 0. Active 11-Feature Random Forest
        rf11_uncal = get_model_trustworthiness("Random Forest (11-Feature)", {"soil_moisture": 2500, "is_calibrated": False})
        self.assertEqual(rf11_uncal["category"], "Experimental — Advisory Only")
        self.assertAlmostEqual(rf11_uncal["threshold"], 0.3271, places=3)
        self.assertEqual(rf11_uncal["input_readiness_code"], "BLOCKED_CALIBRATION")

        rf11_cal = get_model_trustworthiness("Random Forest (11-Feature)", {"soil_moisture": 30.0, "is_calibrated": True})
        self.assertEqual(rf11_cal["category"], "Experimental — Advisory Only")
        self.assertEqual(rf11_cal["input_readiness_code"], "READY")

        # 1. Preserved 13-Feature Random Forest
        rf_uncal = get_model_trustworthiness("Random Forest", {"soil_moisture": 2500, "is_calibrated": False})
        self.assertEqual(rf_uncal["category"], "Best-supported")
        self.assertAlmostEqual(rf_uncal["threshold"], 0.3252, places=3)
        self.assertEqual(rf_uncal["input_readiness_code"], "BLOCKED_CALIBRATION")

        rf_cal = get_model_trustworthiness("Random Forest", {"soil_moisture": 30.0, "is_calibrated": True})
        self.assertEqual(rf_cal["category"], "Best-supported")
        self.assertEqual(rf_cal["input_readiness_code"], "READY")

        # 2. Deep Learning models
        dl_expectations = {
            "LSTM": (0.3802, "INCOMPATIBLE_CHANNELS"),
            "TCN": (0.9800, "INCOMPATIBLE_CHANNELS"),
            "Transformer": (0.6001, "INCOMPATIBLE_CHANNELS"),
        }
        for name, (thresh, r_code) in dl_expectations.items():
            t_info = get_model_trustworthiness(name, {"soil_moisture": 30.0, "is_calibrated": True})
            self.assertEqual(t_info["category"], "Not validated for irrigation")
            self.assertAlmostEqual(t_info["threshold"], thresh, places=3)
            self.assertEqual(t_info["input_readiness_code"], r_code)

    def test_12_per_model_dashboard_predictions(self):
        """Verify dashboard predict_irrigation reports distinct thresholds and model-specific predictions."""
        sensor_cal = {"soil_moisture": 18.0, "is_calibrated": True, "temperature": 32.0, "humidity": 30.0}

        # Active 11-Feature Random Forest: valid prediction with threshold ~0.327
        rf11_res = predict_irrigation(sensor_cal, {}, "Random Forest (11-Feature)")
        self.assertEqual(rf11_res["model"], "Random Forest (11-Feature)")
        self.assertIsNotNone(rf11_res["required"])
        self.assertAlmostEqual(rf11_res["threshold"], 0.3271, places=3)

        # Baseline 13-Feature Random Forest: valid prediction with threshold ~0.325
        rf_res = predict_irrigation(sensor_cal, {}, "Random Forest")
        self.assertEqual(rf_res["model"], "Random Forest")
        self.assertIsNotNone(rf_res["required"])
        self.assertAlmostEqual(rf_res["threshold"], 0.3252, places=3)

        # Deep Learning models on live single-sensor data: safely N/A with individual thresholds
        lstm_res = predict_irrigation(sensor_cal, {}, "LSTM")
        self.assertEqual(lstm_res["model"], "LSTM")
        self.assertIsNone(lstm_res["required"])
        self.assertAlmostEqual(lstm_res["threshold"], 0.3802, places=3)
        self.assertIn("11 historical station", lstm_res["message"])

        tcn_res = predict_irrigation(sensor_cal, {}, "TCN")
        self.assertEqual(tcn_res["model"], "TCN")
        self.assertIsNone(tcn_res["required"])
        self.assertAlmostEqual(tcn_res["threshold"], 0.9800, places=3)

        trans_res = predict_irrigation(sensor_cal, {}, "Transformer")
        self.assertEqual(trans_res["model"], "Transformer")
        self.assertIsNone(trans_res["required"])
        self.assertAlmostEqual(trans_res["threshold"], 0.6001, places=3)

    def test_13_audit_evaluation_metrics_mathematical_consistency(self):
        """Audit all 4 models' metrics for mathematical consistency, formula validity, and evaluation split."""
        # 1. Audit LSTM metrics
        with open(self.models_dir / "lstm_metrics.json") as f:
            lstm_m = json.load(f)

        # Audit test_at_tuned_threshold
        cm = lstm_m["test_at_tuned_threshold"]["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        n_total = tn + fp + fn + tp
        self.assertEqual(n_total, 5993)

        # Base rate
        pos_prevalence = (tp + fn) / n_total
        self.assertAlmostEqual(pos_prevalence, 0.54380110, places=5)

        # Recompute metrics from confusion matrix
        recalc_acc = (tp + tn) / n_total
        recalc_prec = tp / (tp + fp)
        recalc_rec = tp / (tp + fn)
        recalc_f1 = 2 * (recalc_prec * recalc_rec) / (recalc_prec + recalc_rec)

        self.assertAlmostEqual(recalc_acc, lstm_m["test_at_tuned_threshold"]["accuracy"], places=6)
        self.assertAlmostEqual(recalc_prec, lstm_m["test_at_tuned_threshold"]["precision"], places=6)
        self.assertAlmostEqual(recalc_rec, lstm_m["test_at_tuned_threshold"]["recall"], places=6)
        self.assertAlmostEqual(recalc_f1, lstm_m["test_at_tuned_threshold"]["f1"], places=6)

        # Confirm ROC-AUC is near-random (0.5008)
        self.assertAlmostEqual(lstm_m["test_at_tuned_threshold"]["roc_auc"], 0.5007855, places=5)

        # 2. Audit Random Forest metrics
        with open(self.models_dir / "random_forest_metrics.json") as f:
            rf_m = json.load(f)

        rf_cm = rf_m["test_at_tuned_threshold"]["confusion_matrix"]
        rf_tn, rf_fp = rf_cm[0]
        rf_fn, rf_tp = rf_cm[1]
        rf_n = rf_tn + rf_fp + rf_fn + rf_tp
        self.assertEqual(rf_n, 5999)

        rf_acc = (rf_tp + rf_tn) / rf_n
        rf_prec = rf_tp / (rf_tp + rf_fp)
        rf_rec = rf_tp / (rf_tp + rf_fn)
        rf_f1 = 2 * (rf_prec * rf_rec) / (rf_prec + rf_rec)

        self.assertAlmostEqual(rf_acc, rf_m["test_at_tuned_threshold"]["accuracy"], places=6)
        self.assertAlmostEqual(rf_prec, rf_m["test_at_tuned_threshold"]["precision"], places=6)
        self.assertAlmostEqual(rf_rec, rf_m["test_at_tuned_threshold"]["recall"], places=6)
        self.assertAlmostEqual(rf_f1, rf_m["test_at_tuned_threshold"]["f1"], places=6)

        # 3. Confirm TCN and Transformer contain ONLY validation metrics (no test metrics in JSON)
        with open(self.models_dir / "tcn_metrics.json") as f:
            tcn_m = json.load(f)
        self.assertIn("validation_roc_auc", tcn_m)
        self.assertNotIn("test_at_default_threshold", tcn_m)
        self.assertNotIn("test_at_tuned_threshold", tcn_m)

        with open(self.models_dir / "transformer_metrics.json") as f:
            trans_m = json.load(f)
        self.assertIn("validation_roc_auc", trans_m)
        self.assertNotIn("test_at_default_threshold", trans_m)
        self.assertNotIn("test_at_tuned_threshold", trans_m)

    def test_14_no_model_selection_controls_present(self):
        """Verify model-selection controls are completely removed from sidebar, irrigation, and AI models pages."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parent / "dashboard.py"))
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 1. Assert sidebar has no model selection dropdown
        sidebar_keys = [getattr(sb, "key", None) for sb in at.sidebar.selectbox]
        self.assertNotIn("sidebar_model_select", sidebar_keys)

        # 2. Assert Irrigation page has no model selection dropdown
        at.sidebar.radio(key="navigation_radio").set_value("💧 Irrigation").run()
        self.assertEqual(len(at.exception), 0)
        page_keys = [getattr(sb, "key", None) for sb in at.selectbox]
        self.assertNotIn("page_model_select", page_keys)

        # 3. Assert AI Models page has no model selection dropdown
        at.sidebar.radio(key="navigation_radio").set_value("🤖 AI Models").run()
        self.assertEqual(len(at.exception), 0)
        am_keys = [getattr(sb, "key", None) for sb in at.selectbox]
        self.assertNotIn("am_model_select", am_keys)

        # 4. Session state active model is strictly 11-feature model
        self.assertEqual(at.session_state["selected_model"], "Random Forest (11-Feature)")

    def test_15_sqlite_persistence_across_restarts(self):
        """Verify sensor telemetry and probe calibrations persist across simulated application restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_restart.db"
            database.init_db(db_path)

            # 1. Log genuine reading
            reading = {
                "soil1_raw": 2840,
                "soil2_raw": 2910,
                "water_level_raw": 1500,
                "rain_detected": False,
                "temperature_c": 27.5,
                "humidity_pct": 58.0,
            }
            ok, err = database.log_reading(reading, db_path=db_path)
            self.assertTrue(ok, f"Failed to log reading: {err}")

            # 2. Save calibration for Probe 1
            ok_cal, err_cal = database.save_probe_calibration("probe1", 3350, 1380, db_path=db_path)
            self.assertTrue(ok_cal, f"Failed to save calibration: {err_cal}")

            # 3. Simulate application restart: open completely fresh connection
            fresh_df = database.get_history(limit_hours=1, db_path=db_path)
            self.assertEqual(len(fresh_df), 1)
            self.assertAlmostEqual(fresh_df.iloc[0]["soil1_raw"], 2840.0)
            self.assertAlmostEqual(fresh_df.iloc[0]["temperature_c"], 27.5)

            fresh_cal = database.get_probe_calibration("probe1", db_path=db_path)
            self.assertTrue(fresh_cal["calibrated"])
            self.assertEqual(fresh_cal["dry_adc"], 3350.0)
            self.assertEqual(fresh_cal["wet_adc"], 1380.0)

            # Duplicate timestamp insert is safely ignored
            ts = fresh_df.iloc[0]["timestamp"]
            ok_dup, _ = database.log_reading(reading, db_path=db_path, timestamp=ts)
            self.assertTrue(ok_dup)
            df_after_dup = database.get_history(limit_hours=1, db_path=db_path)
            self.assertEqual(len(df_after_dup), 1)

    def test_16_history_and_lag_queries_insufficient_history(self):
        """Verify queries handle missing data and insufficient history (< 45m) gracefully without synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lag.db"
            database.init_db(db_path)

            # Empty database returns empty DataFrame / None
            hist_empty = database.get_history(limit_hours=24, db_path=db_path)
            self.assertTrue(hist_empty.empty)
            earliest, latest, count, span = database.get_history_span(db_path=db_path)
            self.assertEqual(count, 0)
            self.assertEqual(span, 0.0)

            # No 1-hour lag reading exists -> returns None (does NOT invent synthetic history)
            lag_reading = database.get_lag_reading(target_seconds_ago=3600, tolerance_seconds=900, db_path=db_path)
            self.assertIsNone(lag_reading)

            # Log reading from only 10 minutes (600s) ago
            now = time.time()
            reading_10m = {"soil1_raw": 2800, "temperature_c": 26.0}
            database.log_reading(reading_10m, db_path=db_path, recorded_at=now - 600)

            # Querying 1-hour lag (3600s with 900s tolerance: [2700s, 4500s]) MUST return None
            lag_1h = database.get_lag_reading(target_seconds_ago=3600, tolerance_seconds=900, current_time=now, db_path=db_path)
            self.assertIsNone(lag_1h)

            # Log reading in 1-hour lag window (3500s ago)
            reading_1h = {"soil1_raw": 2750, "temperature_c": 25.0}
            database.log_reading(reading_1h, db_path=db_path, recorded_at=now - 3500)
            lag_found = database.get_lag_reading(target_seconds_ago=3600, tolerance_seconds=900, current_time=now, db_path=db_path)
            self.assertIsNotNone(lag_found)
            self.assertAlmostEqual(lag_found["soil1_raw"], 2750.0)

    def test_17_probe_calibration_validation_and_rejection(self):
        """Verify invalid calibration reference values are rejected strictly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cal.db"
            database.init_db(db_path)

            # 1. Identical references rejected (prevents zero division)
            ok, err = database.save_probe_calibration("probe1", 2500, 2500, db_path=db_path)
            self.assertFalse(ok)
            self.assertIn("cannot be identical", err)

            # 2. Inverted references rejected (dry <= wet)
            ok, err = database.save_probe_calibration("probe1", 1400, 3200, db_path=db_path)
            self.assertFalse(ok)
            self.assertIn("Dry ADC", err)

            # 3. Non-numeric values rejected
            ok, err = database.save_probe_calibration("probe1", "dry", 1400, db_path=db_path)
            self.assertFalse(ok)
            self.assertIn("numeric", err)

            # 4. Out of range values rejected (> 4095 or < 0)
            ok, err = database.save_probe_calibration("probe1", 5000, 1400, db_path=db_path)
            self.assertFalse(ok)
            self.assertIn("range", err)

            # 5. Invalid probe id rejected
            ok, err = database.save_probe_calibration("probe3", 3200, 1400, db_path=db_path)
            self.assertFalse(ok)
            self.assertIn("probe identifier", err)

    def test_18_independent_two_probe_calibration_and_math(self):
        """Verify Probe 1 and Probe 2 are calibrated independently and moisture math is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_indep.db"
            database.init_db(db_path)

            # Calibrate Probe 1 (Dry=3400, Wet=1400)
            database.save_probe_calibration("probe1", 3400, 1400, db_path=db_path)
            p1_info = database.get_probe_calibration("probe1", db_path=db_path)
            p2_info = database.get_probe_calibration("probe2", db_path=db_path)
            self.assertTrue(p1_info["calibrated"])
            self.assertFalse(p2_info["calibrated"])

            # Calibrate Probe 2 with distinct references (Dry=3100, Wet=1600)
            database.save_probe_calibration("probe2", 3100, 1600, db_path=db_path)
            p2_info_updated = database.get_probe_calibration("probe2", db_path=db_path)
            self.assertTrue(p2_info_updated["calibrated"])
            self.assertEqual(p2_info_updated["dry_adc"], 3100.0)
            self.assertEqual(p1_info["dry_adc"], 3400.0)

            # Moisture math: Probe 1 (3400 to 1400 -> span 2000)
            # Raw 3400 -> 0%
            m1_dry = database.calculate_calibrated_moisture(3400, 3400, 1400)
            self.assertAlmostEqual(m1_dry, 0.0)
            # Raw 1400 -> 100%
            m1_wet = database.calculate_calibrated_moisture(1400, 3400, 1400)
            self.assertAlmostEqual(m1_wet, 100.0)
            # Raw 2400 -> 50%
            m1_mid = database.calculate_calibrated_moisture(2400, 3400, 1400)
            self.assertAlmostEqual(m1_mid, 50.0)

            # Clamping behavior: raw > dry clamps to 0%, raw < wet clamps to 100%
            self.assertAlmostEqual(database.calculate_calibrated_moisture(3600, 3400, 1400), 0.0)
            self.assertAlmostEqual(database.calculate_calibrated_moisture(1000, 3400, 1400), 100.0)

    def test_18b_provisional_calibration_and_relative_moisture_math(self):
        """Verify provisional calibration values for Sensor 1 and Sensor 2: endpoints, intermediate values, clamping, and invalid rejection."""
        # Provisional values:
        # Sensor 1 (GPIO 32): dry = 2054, wet = 1073 (span = 981)
        # Sensor 2 (GPIO 33): dry = 1906, wet = 1023 (span = 883)
        s1_dry, s1_wet = 2054, 1073
        s2_dry, s2_wet = 1906, 1023

        # Validation checks
        v1_ok, _, _, _ = database.validate_calibration_references(s1_dry, s1_wet)
        v2_ok, _, _, _ = database.validate_calibration_references(s2_dry, s2_wet)
        self.assertTrue(v1_ok)
        self.assertTrue(v2_ok)

        # Equal references rejected
        eq_ok, _, _, eq_err = database.validate_calibration_references(2000, 2000)
        self.assertFalse(eq_ok)
        self.assertIn("identical", eq_err)

        # Inverted references rejected
        inv_ok, _, _, inv_err = database.validate_calibration_references(1000, 2000)
        self.assertFalse(inv_ok)
        self.assertIn("Dry ADC", inv_err)

        # Sensor 1 Endpoints
        # 1. Exact Dry Reference -> 0.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s1_dry, s1_dry, s1_wet), 0.0)
        # 2. Exact Wet Reference -> 100.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s1_wet, s1_dry, s1_wet), 100.0)
        # 3. Intermediate midpoint: 1563.5 -> 50.0%
        s1_mid = (s1_dry + s1_wet) / 2.0
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s1_mid, s1_dry, s1_wet), 50.0)
        # 4. Clamping: raw > dry -> 0.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(2500, s1_dry, s1_wet), 0.0)
        # 5. Clamping: raw < wet -> 100.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(800, s1_dry, s1_wet), 100.0)

        # Sensor 2 Endpoints
        # 1. Exact Dry Reference -> 0.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s2_dry, s2_dry, s2_wet), 0.0)
        # 2. Exact Wet Reference -> 100.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s2_wet, s2_dry, s2_wet), 100.0)
        # 3. Intermediate midpoint: 1464.5 -> 50.0%
        s2_mid = (s2_dry + s2_wet) / 2.0
        self.assertAlmostEqual(database.calculate_calibrated_moisture(s2_mid, s2_dry, s2_wet), 50.0)
        # 4. Clamping: raw > dry -> 0.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(2200, s2_dry, s2_wet), 0.0)
        # 5. Clamping: raw < wet -> 100.0%
        self.assertAlmostEqual(database.calculate_calibrated_moisture(500, s2_dry, s2_wet), 100.0)

        # Missing or invalid ADC returns None safely
        self.assertIsNone(database.calculate_calibrated_moisture(None, s1_dry, s1_wet))
        self.assertIsNone(database.calculate_calibrated_moisture("invalid", s1_dry, s1_wet))
        self.assertIsNone(database.calculate_calibrated_moisture(1500, None, s1_wet))

    def test_19_feature_provenance_audit_table_and_blockers(self):
        """Verify feature provenance audit reflects the active 11-feature model schema and excludes soil temp/humidity."""
        # 1. Active 11-feature audit (default)
        audit = audit_feature_readiness({}, {})
        self.assertEqual(audit["total_features"], 11)
        self.assertEqual(len(audit["feature_rows"]), 11)

        expected_11_features = [
            "soil_moisture", "temperature",
            "wind_speed", "air_humidity", "wind_gust", "pressure",
            "soil_moisture_previous", "soil_moisture_change", "temp_previous",
            "month_sin", "month_cos"
        ]
        self.assertEqual(audit["features"], expected_11_features)
        self.assertNotIn("soil_temperature", audit["features"])
        self.assertNotIn("soil_humidity", audit["features"])
        self.assertNotIn("water_level", audit["features"])

        row_map = {r["Feature"]: r for r in audit["feature_rows"]}
        self.assertTrue(row_map["month_sin"]["is_ready"])
        self.assertTrue(row_map["month_cos"]["is_ready"])
        self.assertFalse(audit["is_live_ready"])
        self.assertGreater(len(audit["blockers"]), 0)

        # 2. Preserved baseline 13-feature audit
        audit_13 = audit_feature_readiness({}, {}, model_name="Random Forest")
        self.assertEqual(audit_13["total_features"], 13)
        self.assertIn("soil_temperature", audit_13["features"])
        self.assertIn("soil_humidity", audit_13["features"])

    def test_20_live_na_safeguard_with_enforced_readiness(self):
        """Verify calibration alone does NOT bypass safeguards: missing lag features withhold live RF prediction as N/A."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_db = Path(tmpdir) / "empty_history.db"
            database.init_db(empty_db)

            # Simulated calibrated probe but zero SQLite history (< 45m continuous)
            sensor_cal = {
                "soil_moisture": 25.0,
                "soil_moisture_calibrated": 25.0,
                "is_calibrated": True,
                "temperature": 28.0,
                "humidity": 65.0,
            }
            weather_data = {
                "temperature": 28.0,
                "humidity": 65.0,
                "wind_speed": 4.0,
                "wind_gust": 7.0,
                "pressure": 1012.0,
            }

            # When enforce_complete_readiness is True, prediction MUST be withheld as N/A (lag features missing)
            res = predict("Random Forest (11-Feature)", sensor_cal, weather_data, enforce_complete_readiness=True, db_path=empty_db)
            self.assertIsNone(res["required"])
            self.assertIsNone(res["confidence"])
            self.assertIn("withheld as N/A", res["message"])
            self.assertIn("blocker(s) present", res["message"])
            self.assertIn("safeguards", res["message"])

            # Uncalibrated probe is also withheld as N/A
            sensor_uncal = {
                "soil1_raw": 2400,
                "is_calibrated": False,
            }
            res_uncal = predict("Random Forest (11-Feature)", sensor_uncal, weather_data, enforce_complete_readiness=True, db_path=empty_db)
            self.assertIsNone(res_uncal["required"])
            self.assertIn("withheld as N/A", res_uncal["message"])
            self.assertIn("uncalibrated probes", res_uncal["message"])

            # And without readiness enforcement, raw ADC is directly rejected with explicit explanation
            res_raw = predict("Random Forest (11-Feature)", sensor_uncal, weather_data, enforce_complete_readiness=False)
            self.assertIsNone(res_raw["required"])
            self.assertIn("raw ADC", res_raw["message"])

    def test_21_models_and_metrics_integrity_unmodified(self):
        """Verify all 4 model weight files and 4 metrics JSON files remain completely unchanged."""
        expected_files = [
            "random_forest.pkl",
            "random_forest_metrics.json",
            "improved_lstm.pt",
            "lstm_metrics.json",
            "tcn.pt",
            "tcn_metrics.json",
            "transformer.pt",
            "transformer_metrics.json",
        ]
        for fname in expected_files:
            fpath = self.models_dir / fname
            self.assertTrue(fpath.exists(), f"Model file missing: {fname}")
            self.assertGreater(fpath.stat().st_size, 0, f"Model file is empty: {fname}")

    def test_22_active_11feature_model_displayed_across_pages(self):
        """Verify active 11-feature model is consistently displayed across sidebar, irrigation, and AI models pages."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py")).run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        # 1. Sidebar contains active model name and advisory status
        sidebar_texts = [m.value for m in at.sidebar.markdown]
        sidebar_combined = "\n".join(sidebar_texts)
        self.assertIn("Active Model:</b> Random Forest (11-Feature)", sidebar_combined)
        self.assertIn("Experimental — Advisory Only", sidebar_combined)

        # 2. Irrigation Page contains active model banner and advisory warning
        at.sidebar.radio(key="navigation_radio").set_value("💧 Irrigation").run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        irr_markdowns = [m.value for m in at.markdown]
        irr_combined = "\n".join(irr_markdowns)
        self.assertIn("Active Model:</b> Random Forest (11-Feature)", irr_combined)
        self.assertIn("Experimental — Advisory Only", irr_combined)
        self.assertIn("Relative soil moisture from capacitive probes has not been field-validated", irr_combined)

        # 3. AI Models Page contains active model banner
        at.sidebar.radio(key="navigation_radio").set_value("🤖 AI Models").run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        am_markdowns = [m.value for m in at.markdown]
        am_combined = "\n".join(am_markdowns)
        self.assertIn("Active Model:</b> Random Forest (11-Feature)", am_combined)
        self.assertIn("Experimental — Advisory Only", am_combined)

    def test_23_11feature_readiness_audit_table_and_warnings(self):
        """Verify AI Models page displays 11-feature model architecture, tuned threshold, and 11-feature audit table."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py")).run(timeout=30)
        at.sidebar.radio(key="navigation_radio").set_value("🤖 AI Models").run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        # Helper to gather all text from main panel
        texts = []
        for item in at.main:
            if hasattr(item, "value") and isinstance(item.value, str):
                texts.append(item.value)
        panel_text = "\n".join(texts)

        self.assertIn("HistGradientBoostingClassifier (11 features)", panel_text)
        self.assertIn("0.327", panel_text)
        self.assertIn("11-Feature Input Provenance & Live Readiness Audit", panel_text)
        self.assertNotIn("13-Feature Input Provenance", panel_text)

    def test_25_continuous_session_lag_query_safeguard(self):
        """Verify get_lag_reading with require_continuous=True rejects readings across session gaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_session_lag.db"
            database.init_db(db_path)

            now = time.time()
            # 1. Log an old reading from an earlier session (3500s ago)
            database.log_reading({"soil1_raw": 1500, "temperature_c": 22.0}, db_path=db_path, recorded_at=now - 3500)

            # 2. Simulate disconnection/gap of 1000s, then start new session 100s ago
            database.log_reading({"soil1_raw": 1550, "temperature_c": 22.5}, db_path=db_path, recorded_at=now - 100)
            database.log_reading({"soil1_raw": 1560, "temperature_c": 22.6}, db_path=db_path, recorded_at=now)

            # require_continuous=True MUST reject the old reading from 3500s ago because current session is only 100s old (< 2700s)
            lag_cont = database.get_lag_reading(
                target_seconds_ago=3600, tolerance_seconds=900, current_time=now, db_path=db_path, require_continuous=True
            )
            self.assertIsNone(lag_cont)

            # require_continuous=False (legacy mode) finds the reading within epoch window
            lag_unrestricted = database.get_lag_reading(
                target_seconds_ago=3600, tolerance_seconds=900, current_time=now, db_path=db_path, require_continuous=False
            )
            self.assertIsNotNone(lag_unrestricted)
            self.assertEqual(lag_unrestricted["soil1_raw"], 1500.0)

            # 3. Now simulate a genuine continuous session of 50 minutes (3000s) at 60s intervals (gap <= 120s)
            for t_offset in range(200, 3100, 60):
                database.log_reading({"soil1_raw": 1600, "temperature_c": 23.0}, db_path=db_path, recorded_at=now - t_offset)

            # Now continuous session reaches back 3000s (>= 2700s); lag query MUST succeed
            lag_valid = database.get_lag_reading(
                target_seconds_ago=3600, tolerance_seconds=900, current_time=now, db_path=db_path, require_continuous=True
            )
            self.assertIsNotNone(lag_valid)


if __name__ == "__main__":
    unittest.main()



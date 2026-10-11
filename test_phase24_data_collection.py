"""
Unit and Integration Tests for Phase 24 Real-World Sensor Data Collection and Calibration Pipeline
====================================================================================================
Tests:
1. Ten-minute interval boundary calculation (floor to 600s boundary).
2. Multiple raw readings within a single 10-minute bucket (mean, std, sample count).
3. Missing DHT22 readings (null handling and valid counts).
4. Empty intervals and non-fabrication of synthetic observations.
5. Long disconnect gaps detection without interpolation.
6. Rain sensor missingness and distinction between presence ratio and mm rainfall depth.
7. Rejection of invalid timestamps (non-finite, NaN).
8. Probe identity preservation in calibration records.
9. Calibration records validation: rejection of missing reference values, invalid units, and out-of-range ADC.
10. Preservation of production 11-feature classifier, tuned threshold, and pump safety lock (PUMP_ALLOWED=False).
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase24"))

from data_collection import (
    aggregate_raw_telemetry,
    identify_aggregation_gaps,
    validate_calibration_record,
    get_14_feature_readiness_audit,
    PUMP_ALLOWED as PUMP_ALLOWED_PHASE24,
)
import training.inference as prod_inference


class TestPhase24DataCollectionPipeline(unittest.TestCase):
    """Test suite for Phase 24 data collection and calibration pipeline."""

    def test_01_ten_minute_interval_boundary_alignment(self):
        """Verify that timestamps are deterministically floored to 600-second interval boundaries."""
        base = 1700000400.0  # Exact multiple of 600 (1700000400 % 600 == 0)
        epochs = [base + 50.0, base + 200.0, base + 599.0, base + 601.0]
        df_raw = pd.DataFrame({
            "recorded_at": epochs,
            "soil1_raw": [1500.0, 1510.0, 1520.0, 1530.0],
            "soil2_raw": [1400.0, 1410.0, 1420.0, 1430.0],
            "temperature_c": [22.0, 22.1, 22.2, 22.5],
            "humidity_pct": [60.0, 60.1, 60.2, 60.5],
            "rain_detected": [0, 0, 0, 0],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        self.assertEqual(len(agg_df), 2)
        # Bucket 1: base to base + 600.0
        self.assertEqual(agg_df["interval_start_epoch"].iloc[0], base)
        self.assertEqual(agg_df["interval_end_epoch"].iloc[0], base + 600.0)
        self.assertEqual(agg_df["raw_sample_count"].iloc[0], 3)
        # Bucket 2: base + 600.0 to base + 1200.0
        self.assertEqual(agg_df["interval_start_epoch"].iloc[1], base + 600.0)
        self.assertEqual(agg_df["raw_sample_count"].iloc[1], 1)

    def test_02_multiple_raw_readings_aggregation_statistics(self):
        """Verify correct mean, std, sample count, and time coverage across multiple readings in one interval."""
        base = 1700000000.0
        # 4 readings spaced at 30 seconds
        epochs = [base + 10.0, base + 40.0, base + 70.0, base + 100.0]
        df_raw = pd.DataFrame({
            "recorded_at": epochs,
            "soil1_raw": [1000.0, 1020.0, 1040.0, 1060.0],
            "soil2_raw": [2000.0, 2010.0, 2020.0, 2030.0],
            "temperature_c": [20.0, 21.0, 22.0, 23.0],
            "humidity_pct": [50.0, 52.0, 54.0, 56.0],
            "rain_detected": [0, 1, 0, 1],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        self.assertEqual(len(agg_df), 1)
        row = agg_df.iloc[0]
        self.assertEqual(row["raw_sample_count"], 4)
        self.assertEqual(row["time_coverage_seconds"], 90.0)  # 100 - 10
        self.assertAlmostEqual(row["soil1_raw_mean"], 1030.0)
        self.assertAlmostEqual(row["temperature_c_mean"], 21.5)
        self.assertAlmostEqual(row["humidity_pct_mean"], 53.0)
        self.assertEqual(row["rain_detected_count"], 2)
        self.assertAlmostEqual(row["rain_presence_ratio"], 0.5)

    def test_03_missing_dht22_readings_handling(self):
        """Verify that null/NaN DHT22 readings are handled cleanly without corrupting other measurements."""
        base = 1700000000.0
        df_raw = pd.DataFrame({
            "recorded_at": [base + 10.0, base + 20.0, base + 30.0],
            "soil1_raw": [1200.0, 1205.0, 1210.0],
            "soil2_raw": [1300.0, 1305.0, 1310.0],
            "temperature_c": [24.0, np.nan, None],
            "humidity_pct": [None, 65.0, np.nan],
            "rain_detected": [0, 0, 0],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        self.assertEqual(len(agg_df), 1)
        row = agg_df.iloc[0]
        self.assertAlmostEqual(row["temperature_c_mean"], 24.0)
        self.assertEqual(row["temperature_c_valid_count"], 1)
        self.assertAlmostEqual(row["humidity_pct_mean"], 65.0)
        self.assertEqual(row["humidity_pct_valid_count"], 1)

    def test_04_empty_intervals_never_synthetically_fabricated(self):
        """Verify that intervals without raw data are not created as synthetic rows."""
        base = 1700000400.0  # multiple of 600
        # Readings in interval 0 and interval 3 (intervals 1 and 2 are empty)
        df_raw = pd.DataFrame({
            "recorded_at": [base + 10.0, base + 1810.0],  # 1810 = 3rd 600s bucket
            "soil1_raw": [1000.0, 1100.0],
            "soil2_raw": [2000.0, 2100.0],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        # Must contain exactly 2 rows, not 4 (no fake rows for empty intervals)
        self.assertEqual(len(agg_df), 2)
        self.assertEqual(agg_df["interval_start_epoch"].iloc[0], base)
        self.assertEqual(agg_df["interval_start_epoch"].iloc[1], base + 1800.0)

    def test_05_disconnect_gaps_detection_without_interpolation(self):
        """Verify identify_aggregation_gaps detects missing intervals and never interpolates."""
        base = 1700000400.0
        df_raw = pd.DataFrame({
            "recorded_at": [base + 10.0, base + 1810.0],
            "soil1_raw": [1000.0, 1100.0],
            "soil2_raw": [2000.0, 2100.0],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        gaps = identify_aggregation_gaps(agg_df, interval_seconds=600.0)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["missing_intervals_count"], 2)  # intervals at 600s and 1200s
        self.assertEqual(gaps[0]["gap_seconds"], 1200.0)

    def test_06_rain_sensor_missingness_and_binary_distinction(self):
        """Verify that interval with no rain data is distinguished from interval with dry indications."""
        base = 1700000400.0
        df_raw = pd.DataFrame({
            "recorded_at": [base + 10.0, base + 610.0],
            "soil1_raw": [1000.0, 1000.0],
            "rain_detected": [0, np.nan],  # Bucket 1 has dry data (0), Bucket 2 has no valid rain data
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        # Bucket 1: valid rain data, ratio 0.0
        self.assertTrue(agg_df["has_valid_rain_data"].iloc[0])
        self.assertEqual(agg_df["rain_presence_ratio"].iloc[0], 0.0)
        # Bucket 2: no valid rain data
        self.assertFalse(agg_df["has_valid_rain_data"].iloc[1])
        self.assertTrue(pd.isna(agg_df["rain_presence_ratio"].iloc[1]))

    def test_07_invalid_timestamps_handling(self):
        """Verify that rows with NaN or infinite timestamps are dropped safely."""
        df_raw = pd.DataFrame({
            "recorded_at": [1700000010.0, np.nan, np.inf, 1700000020.0],
            "soil1_raw": [1000.0, 1050.0, 1060.0, 1020.0],
            "soil2_raw": [2000.0, 2050.0, 2060.0, 2020.0],
        })
        agg_df = aggregate_raw_telemetry(df_raw, interval_seconds=600.0)
        self.assertEqual(len(agg_df), 1)
        self.assertEqual(agg_df["raw_sample_count"].iloc[0], 2)

    def test_08_calibration_record_validation_success(self):
        """Verify valid calibration record is accepted and sanitized properly."""
        valid_record = {
            "probe_id": "probe1",
            "raw_adc": 1540.0,
            "timestamp": "2026-10-10T12:00:00Z",
            "sample_id": "CORE_01_SATURATED",
            "reference_value": 42.5,
            "reference_unit": "vwc_pct",
            "soil_type": "Sandy Loam",
            "bulk_density_g_cm3": 1.35,
            "sample_temperature_c": 21.0,
            "calibration_method": "gravimetric_oven_dry",
            "notes": "Laboratory test core 1",
        }
        is_valid, sanitized, err = validate_calibration_record(valid_record)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(sanitized["probe_id"], "probe1")
        self.assertEqual(sanitized["raw_adc"], 1540.0)
        self.assertEqual(sanitized["reference_value"], 42.5)
        self.assertEqual(sanitized["reference_unit"], "vwc_pct")

    def test_09_calibration_record_rejection_rules(self):
        """Verify strict rejection of missing fields, invalid units, and out-of-range ADC."""
        # 1. Invalid probe id
        ok, _, err = validate_calibration_record({"probe_id": "probe3"})
        self.assertFalse(ok)
        self.assertIn("Invalid probe_id", err)

        # 2. Out of range ADC count
        ok, _, err = validate_calibration_record({
            "probe_id": "probe1", "raw_adc": 5000.0, "timestamp": 1700000000.0,
            "sample_id": "C1", "reference_value": 20.0, "reference_unit": "vwc_pct"
        })
        self.assertFalse(ok)
        self.assertIn("out of 12-bit ADC range", err)

        # 3. Missing reference value
        ok, _, err = validate_calibration_record({
            "probe_id": "probe1", "raw_adc": 1500.0, "timestamp": 1700000000.0,
            "sample_id": "C1", "reference_unit": "vwc_pct"
        })
        self.assertFalse(ok)
        self.assertIn("Missing mandatory field 'reference_value'", err)

        # 4. Invalid reference unit
        ok, _, err = validate_calibration_record({
            "probe_id": "probe1", "raw_adc": 1500.0, "timestamp": 1700000000.0,
            "sample_id": "C1", "reference_value": 20.0, "reference_unit": "volts"
        })
        self.assertFalse(ok)
        self.assertIn("Invalid reference_unit", err)

    def test_10_production_invariance_and_safety_lock(self):
        """Verify production classifier, tuned threshold, and pump safety lock remain locked."""
        self.assertFalse(PUMP_ALLOWED_PHASE24, "Phase 24 PUMP_ALLOWED must be False")
        self.assertEqual(prod_inference.ACTIVE_MODEL_NAME, "Random Forest (11-Feature)")
        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)
        self.assertEqual(len(metrics["features"]), 11)


if __name__ == "__main__":
    unittest.main()

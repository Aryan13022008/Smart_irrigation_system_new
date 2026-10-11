"""
Unit and Integration Tests for Phase 23 Live Telemetry Sequence Buffer
======================================================================
Tests:
1. Valid 24-observation sequence construction under nominal 10-minute cadence.
2. Strict rejection of fewer than 24 observations.
3. Strict rejection of irregular temporal spacing and excessive gaps.
4. Strict rejection of unordered (non-monotonic) timestamps.
5. Strict rejection of mixed sensor/probe identities.
6. Strict rejection of missing feature columns and incompatible orders.
7. Strict rejection of non-finite (NaN / Inf) values.
8. Strict rejection of high-frequency (2.5s) serial packet streams masquerading as 10m steps.
9. Verification that no future rows enter the sequence.
10. Preservation of production Phase 11 classifier and PUMP_ALLOWED = False safety lock.
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
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase23"))

from sequence_buffer import (
    validate_and_construct_sequence,
    audit_live_telemetry_history,
    EXPERIMENTAL_14_FEATURES,
    EXPECTED_SEQUENCE_LENGTH,
    PUMP_ALLOWED as PUMP_ALLOWED_PHASE23,
)
import training.inference as prod_inference


class TestPhase23SequenceBuffer(unittest.TestCase):
    """Test suite for Phase 23 sequence buffer validation and safety."""

    def setUp(self):
        # Generate a valid reference 24-step 10-minute sequence
        base_epoch = 1700000000.0  # arbitrary epoch
        epochs = [base_epoch + i * 600.0 for i in range(24)]
        timestamps = [datetime.fromtimestamp(e, tz=timezone.utc).isoformat() for e in epochs]

        data = {
            "recorded_at": epochs,
            "timestamp": timestamps,
            "probe_id": ["probe1"] * 24,
            "weather_temp": np.linspace(20.0, 25.0, 24),
            "weather_humidity": np.linspace(60.0, 50.0, 24),
            "weather_rain": [0.0] * 24,
            "weather_pressure": [1013.25] * 24,
            "weather_wind_speed": [2.5] * 24,
            "weather_radiation": [350.0] * 24,
            "soil_temperature_0-7cm": [18.5] * 24,
            "soil_temperature_7-18cm": [17.8] * 24,
            "ec": [1.2] * 24,
            "ph": [6.8] * 24,
            "soil_moisture": np.linspace(35.0, 32.0, 24),
            "irrigation_duration_minutes": [0.0] * 24,
            "hour": [(i % 24) for i in range(24)],
            "month": [10] * 24,
        }
        self.valid_df = pd.DataFrame(data)

    def test_01_valid_sequence_passes(self):
        """Verify that a genuine 24-step 10-minute sequence is validated successfully."""
        res = validate_and_construct_sequence(self.valid_df)
        self.assertTrue(res.is_valid)
        self.assertIsNotNone(res.sequence)
        self.assertEqual(len(res.sequence), 24)
        self.assertEqual(list(res.sequence.columns), EXPERIMENTAL_14_FEATURES)
        self.assertEqual(res.diagnostics["n_steps"], 24)
        self.assertEqual(res.diagnostics["total_span_seconds"], 23 * 600.0)

    def test_02_reject_fewer_than_24_observations(self):
        """Verify that sequences with fewer than 24 observations are strictly rejected."""
        short_df = self.valid_df.iloc[:20].copy()
        res = validate_and_construct_sequence(short_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "INSUFFICIENT_OBSERVATIONS")
        self.assertIn("Insufficient observations: found 20 rows", res.blocker_message)

    def test_03_reject_irregular_spacing_and_gaps(self):
        """Verify that timestamp gaps exceeding tolerance (e.g. 15-minute gap) are rejected."""
        gap_df = self.valid_df.copy()
        # Introduce a 20-minute gap at step 10
        gap_df.loc[10:, "recorded_at"] += 600.0
        res = validate_and_construct_sequence(gap_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "IRREGULAR_TEMPORAL_SPACING")
        self.assertIn("violate expected 10-minute cadence", res.blocker_message)

    def test_04_reject_unordered_and_duplicate_timestamps(self):
        """Verify that non-monotonic or duplicate timestamps are strictly rejected."""
        unordered_df = self.valid_df.copy()
        # Swap rows 5 and 6
        unordered_df.loc[5, "recorded_at"], unordered_df.loc[6, "recorded_at"] = (
            unordered_df.loc[6, "recorded_at"],
            unordered_df.loc[5, "recorded_at"],
        )
        res = validate_and_construct_sequence(unordered_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "NON_MONOTONIC_TIMESTAMPS")

        # Duplicate timestamp
        duplicate_df = self.valid_df.copy()
        duplicate_df.loc[5, "recorded_at"] = duplicate_df.loc[4, "recorded_at"]
        res_dup = validate_and_construct_sequence(duplicate_df)
        self.assertFalse(res_dup.is_valid)
        self.assertEqual(res_dup.blocker_code, "NON_MONOTONIC_TIMESTAMPS")

    def test_05_reject_mixed_probe_identities(self):
        """Verify that mixing readings from probe1 and probe2 in one sequence is rejected."""
        mixed_df = self.valid_df.copy()
        mixed_df.loc[12:, "probe_id"] = "probe2"
        res = validate_and_construct_sequence(mixed_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "MIXED_PROBE_IDENTITIES")
        self.assertIn("Sequence mixes measurements from multiple probe identities", res.blocker_message)

    def test_06_reject_missing_feature_columns(self):
        """Verify that missing model features are rejected without synthetic fabrication."""
        missing_df = self.valid_df.drop(columns=["weather_pressure", "ec"]).copy()
        res = validate_and_construct_sequence(missing_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "MISSING_FEATURE_COLUMNS")
        self.assertIn("weather_pressure", res.diagnostics["missing_features"])
        self.assertIn("ec", res.diagnostics["missing_features"])

    def test_07_reject_non_finite_values(self):
        """Verify that NaN or infinite sensor values trigger strict rejection."""
        nan_df = self.valid_df.copy()
        nan_df.loc[5, "soil_moisture"] = np.nan
        res = validate_and_construct_sequence(nan_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "NON_FINITE_FEATURE_VALUES")
        self.assertIn("soil_moisture", res.diagnostics["nan_features"])

    def test_08_reject_high_frequency_packet_stream(self):
        """Verify that 2.5-second ESP32 serial loops are not mistaken for 10-minute sequence steps."""
        base_epoch = 1700000000.0
        fast_epochs = [base_epoch + i * 2.5 for i in range(24)]
        fast_df = self.valid_df.copy()
        fast_df["recorded_at"] = fast_epochs
        res = validate_and_construct_sequence(fast_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.blocker_code, "HIGH_FREQUENCY_PACKET_STREAM")
        self.assertIn("minimum step = 2.5s", res.blocker_message)

    def test_09_no_future_rows_and_trailing_window_alignment(self):
        """Verify that when presented with a large dataset, only past/trailing rows are selected."""
        # Create 50 rows
        base_epoch = 1700000000.0
        epochs = [base_epoch + i * 600.0 for i in range(50)]
        long_data = {
            "recorded_at": epochs,
            "probe_id": ["probe1"] * 50,
            **{f: [1.0] * 50 for f in EXPERIMENTAL_14_FEATURES},
        }
        long_df = pd.DataFrame(long_data)
        res = validate_and_construct_sequence(long_df)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.sequence), 24)
        # Verify it selected the trailing window ending at epoch index 49
        self.assertEqual(res.diagnostics["end_epoch"], base_epoch + 49 * 600.0)
        self.assertEqual(res.diagnostics["start_epoch"], base_epoch + (49 - 23) * 600.0)

    def test_10_production_invariance_and_safety_lock(self):
        """Verify that production 11-feature model, threshold, and pump safety lock remain locked."""
        self.assertFalse(PUMP_ALLOWED_PHASE23, "Phase 23 PUMP_ALLOWED must be strictly False")
        self.assertEqual(prod_inference.ACTIVE_MODEL_NAME, "Random Forest (11-Feature)")
        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)
        self.assertEqual(len(metrics["features"]), 11)


if __name__ == "__main__":
    unittest.main()

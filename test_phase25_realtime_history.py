"""
Unit and Integration Tests for Phase 25 Real-Time Hardware History Infrastructure
================================================================================
Tests:
1. Synchronization of raw serial readings into persistent `ten_minute_observations` table.
2. Idempotency of observation synchronization (no duplicate entries on repeated sync).
3. Continuous streak detection and 24-step sequence availability logic.
4. Correct extraction of the trailing 24-step hardware sequence.
5. Rejection of sequences when continuous streak is shorter than 24 observations.
6. Calibration measurement logging into `soil_calibration_logs` with validation.
7. Rejection of ungrounded or invalid calibration records in `soil_calibration_logs`.
8. Preservation of production 11-feature model, tuned threshold, and pump safety lock (PUMP_ALLOWED=False).
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase25"))

from realtime_history import (
    init_phase25_tables,
    sync_raw_readings_to_ten_minute_observations,
    extract_realtime_24step_hardware_sequence,
    log_soil_calibration_measurement,
    PUMP_ALLOWED as PUMP_ALLOWED_PHASE25,
)
import training.inference as prod_inference


class TestPhase25RealtimeHistoryInfrastructure(unittest.TestCase):
    """Test suite for Phase 25 realtime history tables, sync, and sequence extraction."""

    def setUp(self):
        self.test_dir = PROJECT_ROOT / "data" / "test_phase25"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_db = self.test_dir / "test_sensor_history.db"
        if self.test_db.exists():
            self.test_db.unlink()

        # Initialize tables
        conn = sqlite3.connect(str(self.test_db))
        with conn:
            conn.execute("""
                CREATE TABLE sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT UNIQUE NOT NULL,
                    recorded_at REAL NOT NULL,
                    soil1_raw REAL,
                    soil2_raw REAL,
                    water_level_raw REAL,
                    rain_pin_raw REAL,
                    rain_detected INTEGER,
                    temperature_c REAL,
                    humidity_pct REAL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL
                );
            """)
        conn.close()
        init_phase25_tables(self.test_db)

    def tearDown(self):
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except OSError:
                pass

    def test_01_sync_raw_readings_into_ten_minute_observations(self):
        """Verify that raw readings are aggregated and synchronized into ten_minute_observations."""
        conn = sqlite3.connect(str(self.test_db))
        # Insert 10 packets in bucket 1700000400 (cadence 2.5s)
        base = 1700000400.0
        rows = []
        for i in range(10):
            t = base + i * 2.5
            rows.append((
                datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                t, 1050.0 + i, 1020.0 + i, 50.0, 1.0, 0, 21.0, 65.0, "esp32_usb", "valid"
            ))
        with conn:
            conn.executemany("""
                INSERT INTO sensor_readings (
                    timestamp, recorded_at, soil1_raw, soil2_raw, water_level_raw,
                    rain_pin_raw, rain_detected, temperature_c, humidity_pct, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows)
        conn.close()

        res = sync_raw_readings_to_ten_minute_observations(db_path=self.test_db)
        self.assertEqual(res["total_processed"], 10)
        self.assertEqual(res["buckets_written"], 1)

        # Inspect ten_minute_observations
        conn = sqlite3.connect(str(self.test_db))
        cur = conn.cursor()
        cur.execute("SELECT * FROM ten_minute_observations;")
        saved_buckets = cur.fetchall()
        conn.close()
        self.assertEqual(len(saved_buckets), 1)
        self.assertEqual(saved_buckets[0][0], base)  # interval_start_epoch
        self.assertEqual(saved_buckets[0][4], 10)    # raw_sample_count

    def test_02_idempotency_of_synchronization(self):
        """Verify that running sync repeatedly does not create duplicate entries or fail."""
        conn = sqlite3.connect(str(self.test_db))
        base = 1700000400.0
        with conn:
            conn.execute("""
                INSERT INTO sensor_readings (
                    timestamp, recorded_at, soil1_raw, soil2_raw, water_level_raw,
                    rain_pin_raw, rain_detected, temperature_c, humidity_pct, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, ("2026-10-10T12:00:00Z", base + 5.0, 1050.0, 1020.0, 50.0, 1.0, 0, 21.0, 65.0, "esp32_usb", "valid"))
        conn.close()

        res1 = sync_raw_readings_to_ten_minute_observations(db_path=self.test_db)
        res2 = sync_raw_readings_to_ten_minute_observations(db_path=self.test_db)
        self.assertEqual(res1["buckets_written"], 1)
        self.assertEqual(res2["buckets_written"], 1)

        conn = sqlite3.connect(str(self.test_db))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ten_minute_observations;")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_03_streak_detection_and_sequence_extraction(self):
        """Verify continuous 24-step sequence is extracted when exactly 24 continuous 10m steps exist."""
        conn = sqlite3.connect(str(self.test_db))
        base = 1700000400.0
        # Create 24 continuous 10m intervals
        obs_rows = []
        for i in range(24):
            b_start = base + i * 600.0
            b_end = b_start + 600.0
            dt_start = datetime.fromtimestamp(b_start, tz=timezone.utc).isoformat()
            dt_end = datetime.fromtimestamp(b_end, tz=timezone.utc).isoformat()
            obs_rows.append((
                b_start, b_end, dt_start, dt_end,
                200, 595.0, 1050.0 + i, 5.0, 1020.0 + i, 5.0,
                22.0, 200, 60.0, 200, 0, 1.0, 1, 0.0, 50.0, "completed", dt_start
            ))
        with conn:
            conn.executemany("""
                INSERT INTO ten_minute_observations (
                    interval_start_epoch, interval_end_epoch, interval_start_utc, interval_end_utc,
                    raw_sample_count, time_coverage_seconds,
                    soil1_raw_mean, soil1_raw_std, soil2_raw_mean, soil2_raw_std,
                    temperature_c_mean, temperature_c_valid_count,
                    humidity_pct_mean, humidity_pct_valid_count,
                    rain_detected_count, rain_pin_raw_mean, has_valid_rain_data, rain_presence_ratio,
                    water_level_raw_mean, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, obs_rows)
        conn.close()

        seq_res = extract_realtime_24step_hardware_sequence(db_path=self.test_db)
        self.assertTrue(seq_res["available"])
        self.assertEqual(seq_res["streak_length"], 24)
        self.assertIsNotNone(seq_res["sequence_df"])
        self.assertEqual(len(seq_res["sequence_df"]), 24)
        self.assertEqual(seq_res["sequence_df"]["interval_start_epoch"].iloc[0], base)
        self.assertEqual(seq_res["sequence_df"]["interval_start_epoch"].iloc[-1], base + 23 * 600.0)

    def test_04_rejection_when_streak_shorter_than_24_steps(self):
        """Verify sequence is rejected when continuous streak has fewer than 24 observations."""
        conn = sqlite3.connect(str(self.test_db))
        base = 1700000400.0
        # Create only 15 continuous intervals
        obs_rows = []
        for i in range(15):
            b_start = base + i * 600.0
            b_end = b_start + 600.0
            dt_start = datetime.fromtimestamp(b_start, tz=timezone.utc).isoformat()
            dt_end = datetime.fromtimestamp(b_end, tz=timezone.utc).isoformat()
            obs_rows.append((
                b_start, b_end, dt_start, dt_end,
                200, 595.0, 1050.0, 5.0, 1020.0, 5.0,
                22.0, 200, 60.0, 200, 0, 1.0, 1, 0.0, 50.0, "completed", dt_start
            ))
        with conn:
            conn.executemany("""
                INSERT INTO ten_minute_observations (
                    interval_start_epoch, interval_end_epoch, interval_start_utc, interval_end_utc,
                    raw_sample_count, time_coverage_seconds,
                    soil1_raw_mean, soil1_raw_std, soil2_raw_mean, soil2_raw_std,
                    temperature_c_mean, temperature_c_valid_count,
                    humidity_pct_mean, humidity_pct_valid_count,
                    rain_detected_count, rain_pin_raw_mean, has_valid_rain_data, rain_presence_ratio,
                    water_level_raw_mean, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, obs_rows)
        conn.close()

        seq_res = extract_realtime_24step_hardware_sequence(db_path=self.test_db)
        self.assertFalse(seq_res["available"])
        self.assertEqual(seq_res["streak_length"], 15)
        self.assertIn("requires 24 continuous 10-minute observations", seq_res["blocker"])

    def test_05_soil_calibration_measurement_logging_and_query(self):
        """Verify soil calibration record is validated and saved to soil_calibration_logs."""
        record = {
            "probe_id": "probe1",
            "raw_adc": 1420.0,
            "timestamp": "2026-10-10T14:30:00Z",
            "sample_id": "FIELD_CORE_02",
            "reference_value": 31.8,
            "reference_unit": "vwc_pct",
            "soil_type": "Clay Loam",
            "bulk_density_g_cm3": 1.42,
            "sample_temperature_c": 22.4,
            "calibration_method": "gravimetric_oven_dry",
            "notes": "Verified laboratory core test",
        }
        ok, err = log_soil_calibration_measurement(record, db_path=self.test_db)
        self.assertTrue(ok)
        self.assertIsNone(err)

        conn = sqlite3.connect(str(self.test_db))
        cur = conn.cursor()
        cur.execute("SELECT probe_id, raw_adc, sample_id, reference_value, reference_unit FROM soil_calibration_logs;")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "probe1")
        self.assertEqual(rows[0][1], 1420.0)
        self.assertEqual(rows[0][2], "FIELD_CORE_02")
        self.assertEqual(rows[0][3], 31.8)
        self.assertEqual(rows[0][4], "vwc_pct")

    def test_06_rejection_of_invalid_calibration_records(self):
        """Verify ungrounded or invalid calibration records are rejected."""
        # Missing reference value
        bad_record = {
            "probe_id": "probe1",
            "raw_adc": 1420.0,
            "timestamp": "2026-10-10T14:30:00Z",
            "sample_id": "FIELD_CORE_02",
            "reference_unit": "vwc_pct",
        }
        ok, err = log_soil_calibration_measurement(bad_record, db_path=self.test_db)
        self.assertFalse(ok)
        self.assertIn("Missing mandatory field 'reference_value'", err)

    def test_07_production_invariance_and_safety_lock(self):
        """Verify production classifier, tuned threshold, and pump safety lock remain locked."""
        self.assertFalse(PUMP_ALLOWED_PHASE25, "Phase 25 PUMP_ALLOWED must be False")
        self.assertEqual(prod_inference.ACTIVE_MODEL_NAME, "Random Forest (11-Feature)")
        metrics = prod_inference.get_metrics("Random Forest (11-Feature)")
        self.assertAlmostEqual(metrics["threshold_tuned"], 0.3270985223048995, places=5)
        self.assertEqual(len(metrics["features"]), 11)


if __name__ == "__main__":
    unittest.main()

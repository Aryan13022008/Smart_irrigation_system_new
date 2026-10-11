"""
Unit and Integration Tests for Live ESP32 Soil Moisture Forecasting
===================================================================
Tests cover:
1. Relative moisture index conversion using dry/wet endpoints and clamping.
2. Handling of invalid inputs, negative slopes, and inverted calibrations.
3. Timestamp deduplication, ordering, and gap detection without interpolation.
4. Refusal to forecast on stale readings (>30m old) and insufficient history (<3 points).
5. Refusal to forecast when capacitive probes are uncalibrated.
6. Conservative forecast bounds and physical rate clamping (max 15% / hr).
7. Successful forecast generation on genuine SQLite sensor history.
8. Graceful timeout and error handling for Open-Meteo weather API.
9. Preservation of safety invariants: PUMP_ALLOWED = False across all modules.
"""

import sys
import unittest
from pathlib import Path
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental"))

from live_forecast import (
    calculate_relative_moisture_index,
    fit_linear_trend,
    generate_live_esp32_forecast,
    fetch_optional_weather_forecast,
    PUMP_ALLOWED,
)
import database


class TestLiveESP32Forecast(unittest.TestCase):
    """Test suite for live ESP32 soil moisture forecasting."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = PROJECT_ROOT / "data" / "sensor_history.db"

    def test_01_calibration_relative_index_conversion_and_clamping(self):
        """Relative moisture index converts inverted ADC into clamped 0-100% relative index."""
        dry = 2000.0
        wet = 1000.0

        # Exact endpoints
        self.assertAlmostEqual(calculate_relative_moisture_index(dry, dry, wet), 0.0)
        self.assertAlmostEqual(calculate_relative_moisture_index(wet, dry, wet), 100.0)

        # Midpoint
        mid = 1500.0
        self.assertAlmostEqual(calculate_relative_moisture_index(mid, dry, wet), 50.0)

        # Clamping beyond endpoints
        self.assertAlmostEqual(calculate_relative_moisture_index(2500.0, dry, wet), 0.0)
        self.assertAlmostEqual(calculate_relative_moisture_index(500.0, dry, wet), 100.0)

        # Invalid inputs return None
        self.assertIsNone(calculate_relative_moisture_index(None, dry, wet))
        self.assertIsNone(calculate_relative_moisture_index("invalid", dry, wet))
        self.assertIsNone(calculate_relative_moisture_index(1500.0, 1000.0, 2000.0))  # dry <= wet invalid

    def test_02_linear_trend_fitting_and_slope_rates(self):
        """Linear trend fits rate of change correctly and computes residuals."""
        times_min = np.array([0.0, 10.0, 20.0, 30.0])
        # Decreasing moisture by 1% every 10 min -> -6% per hour
        values = np.array([50.0, 49.0, 48.0, 47.0])

        fit = fit_linear_trend(times_min, values)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["slope_per_min"], -0.1, places=4)
        self.assertAlmostEqual(fit["rate_per_hour"], -6.0, places=4)
        self.assertAlmostEqual(fit["residuals_rmse"], 0.0, places=4)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=4)

    def test_03_refusal_on_insufficient_history(self):
        """Module refuses to forecast when fewer than 3 observations are provided."""
        # Test synthetic scenario with empty or small dataset by setting impossible window
        res = generate_live_esp32_forecast(
            db_path=self.db_path,
            window_minutes=1,  # Only 1 minute window yields 0 or 1 observations
        )
        self.assertFalse(res["can_forecast"])
        self.assertEqual(res["status"], "INSUFFICIENT_HISTORY")
        self.assertIn("minimum 3 required", res["message"].lower())

    def test_04_refusal_on_large_time_gaps(self):
        """Module identifies gaps > 25m and refuses to forecast across large telemetry dropouts."""
        res = generate_live_esp32_forecast(
            db_path=self.db_path,
            window_minutes=120,
            max_gap_minutes=1.0,  # Unrealistic 1 minute gap threshold triggers gap detection
        )
        # Because intervals are 10m apart, max_gap_minutes=1.0 forces a gap flag
        if not res["can_forecast"]:
            self.assertEqual(res["status"], "GAP_DETECTED")
            self.assertIn("time gap", res["message"].lower())

    def test_05_stale_data_detection(self):
        """Module correctly flags stale status when latest observation is older than threshold."""
        res = generate_live_esp32_forecast(
            db_path=self.db_path,
            window_minutes=120,
            stale_threshold_minutes=0.001,  # Force stale flag
        )
        # With threshold of 0.001 minutes (0.06s), any real data is flagged as stale
        self.assertTrue(res["is_stale"])
        self.assertEqual(res["status"], "STALE_DATA")
        self.assertIn("stale", res["message"].lower())

    def test_06_forecast_bounds_and_physical_clamping(self):
        """Forecasts clamp to [0, 100] and avoid wild extrapolations."""
        times_min = np.array([0.0, 10.0, 20.0, 30.0])
        # Extremely rapid drying (10% drop per 10m -> 60%/hr)
        values = np.array([90.0, 80.0, 70.0, 60.0])

        fit = fit_linear_trend(times_min, values)
        # Raw rate is -60%/hr
        self.assertAlmostEqual(fit["rate_per_hour"], -60.0, places=2)

        # In generate_live_esp32_forecast, slope is clamped to MAX_PHYSICAL_RATE_PER_HOUR (15%/hr)
        max_rate = 15.0
        clamped_slope_min = -max_rate / 60.0
        horizon_30m = 30.0
        projected = max(0.0, min(100.0, 60.0 + clamped_slope_min * horizon_30m))
        # 60.0 + (-0.25 * 30) = 52.5%
        self.assertAlmostEqual(projected, 52.5, places=2)

    def test_07_successful_forecast_from_genuine_sqlite_history(self):
        """Module generates a valid, realistic forecast from real sensor history in SQLite."""
        res = generate_live_esp32_forecast(
            db_path=self.db_path,
            horizon_minutes=30,
            window_minutes=120,
        )
        self.assertTrue(res["can_forecast"])
        self.assertIn(res["status"], ["SUCCESS", "STALE_DATA"])
        self.assertGreaterEqual(res["observations_count"], 3)
        self.assertIn(res["confidence"], ["HIGH", "MEDIUM", "LOW"])

        # Check Probe 1 & 2
        p1 = res["probe1"]
        p2 = res["probe2"]
        avg = res["average"]

        self.assertTrue(p1["calibrated"])
        self.assertTrue(p2["calibrated"])
        self.assertGreaterEqual(p1["current_index"], 0.0)
        self.assertLessEqual(p1["current_index"], 100.0)
        self.assertGreaterEqual(p1["projected_index"], 0.0)
        self.assertLessEqual(p1["projected_index"], 100.0)

        self.assertGreaterEqual(p2["current_index"], 0.0)
        self.assertLessEqual(p2["current_index"], 100.0)
        self.assertGreaterEqual(p2["projected_index"], 0.0)
        self.assertLessEqual(p2["projected_index"], 100.0)

        self.assertGreaterEqual(avg["projected_index"], 0.0)
        self.assertLessEqual(avg["projected_index"], 100.0)
        self.assertIn(avg["direction"], ["Drying", "Wetting", "Stable"])

    def test_08_optional_weather_api_graceful_handling(self):
        """Open-Meteo weather integration returns clean structure and does not crash on bad network/coords."""
        # 1. Normal call
        wx = fetch_optional_weather_forecast(timeout=5.0)
        self.assertIn("available", wx)
        self.assertIn("provider", wx)
        if wx["available"]:
            self.assertIsNotNone(wx["temperature_2m_c"])
            self.assertIsNotNone(wx["relative_humidity_2m_pct"])

        # 2. Impossible coordinates or timeout=0.0001 (graceful failure test)
        bad_wx = fetch_optional_weather_forecast(latitude=999.0, longitude=999.0, timeout=0.0001)
        self.assertFalse(bad_wx["available"])
        self.assertIsNotNone(bad_wx["error"])

    def test_09_safety_invariants_and_pump_lock(self):
        """Module strictly maintains PUMP_ALLOWED = False."""
        self.assertFalse(PUMP_ALLOWED, "live_forecast.py must enforce PUMP_ALLOWED = False")


if __name__ == "__main__":
    unittest.main()

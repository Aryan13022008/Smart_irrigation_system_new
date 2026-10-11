"""
Focused Unit Tests for External Weather Integration & 14-Feature Compatibility
=============================================================================
Validates:
1. Open-Meteo hourly weather query parsing, required feature schema, and unit matching:
   - surface_pressure -> weather_pressure (hPa)
   - wind_speed_10m -> weather_wind_speed (km/h)
   - direct_radiation -> weather_radiation (W/m²)
   - soil_temperature_0_to_7cm -> soil_temperature_0-7cm (°C)
2. In-memory caching logic avoiding redundant API requests.
3. Timestamp alignment using linear interpolation on UTC index with strict boundary enforcement
   (rejection of extrapolation beyond weather timestamp bounds).
4. Graceful handling of network timeouts, API failures, and unconfigured coordinates
   without crashing the live trend forecasting module.
5. Strict zero-fabrication schema guard: confirms that incomplete live sequences lacking the
   4 in-situ channels (subsoil temperature, EC, pH, irrigation duration) raise ValueError
   and are safely withheld from neural network inference.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase13"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "experimental" / "phase15_transformer"))

from live_forecast import (
    fetch_hourly_weather_observations,
    fetch_optional_weather_forecast,
    align_weather_to_observation_timestamps,
    _WEATHER_CACHE,
    PUMP_ALLOWED,
)
from forecasting_models import (
    load_forecasting_model,
    predict_future_moisture,
    preprocess_sequence,
    EXPERIMENTAL_FEATURES,
)


class TestExternalWeatherIntegration(unittest.TestCase):
    """Test suite for weather integration, alignment, and schema guards."""

    def setUp(self):
        _WEATHER_CACHE.clear()

    def test_01_weather_hourly_mock_parsing_and_units(self):
        """Validates that hourly weather parsing extracts exact columns and units."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_times = [
            (datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
            for i in range(5)
        ]
        mock_response.json.return_value = {
            "hourly": {
                "time": mock_times,
                "surface_pressure": [1013.2, 1013.0, 1012.8, 1012.5, 1012.3],
                "wind_speed_10m": [12.4, 11.8, 10.5, 9.2, 8.7],
                "direct_radiation": [250.0, 210.0, 150.0, 80.0, 0.0],
                "soil_temperature_0_to_7cm": [22.4, 22.1, 21.8, 21.4, 20.9],
                "soil_temperature_7_to_28cm": [20.1, 20.1, 20.2, 20.2, 20.1],
            },
            "hourly_units": {
                "surface_pressure": "hPa",
                "wind_speed_10m": "km/h",
                "direct_radiation": "W/m²",
                "soil_temperature_0_to_7cm": "°C",
            },
        }

        with patch("requests.get", return_value=mock_response):
            res = fetch_hourly_weather_observations(latitude=12.9184, longitude=79.1325)

        self.assertTrue(res["available"])
        self.assertIsNone(res["error"])
        df_w = res["weather_df"]
        self.assertIsNotNone(df_w)
        self.assertEqual(len(df_w), 5)

        for col in ["weather_pressure", "weather_wind_speed", "weather_radiation", "soil_temperature_0-7cm"]:
            self.assertIn(col, df_w.columns)
            self.assertTrue(np.all(np.isfinite(df_w[col])))

        self.assertEqual(res["units"]["surface_pressure"], "hPa")
        self.assertEqual(res["units"]["wind_speed_10m"], "km/h")
        self.assertEqual(res["units"]["direct_radiation"], "W/m²")

    def test_02_weather_caching_behavior(self):
        """Validates that second identical call reuses cached dictionary without network hit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2026-10-10T12:00:00+00:00"],
                "surface_pressure": [1012.0],
                "wind_speed_10m": [10.0],
                "direct_radiation": [200.0],
                "soil_temperature_0_to_7cm": [22.0],
                "soil_temperature_7_to_28cm": [20.0],
            },
            "hourly_units": {"surface_pressure": "hPa"},
        }

        with patch("requests.get", return_value=mock_response) as mock_get:
            res1 = fetch_hourly_weather_observations(latitude=12.9184, longitude=79.1325)
            self.assertEqual(mock_get.call_count, 1)

            res2 = fetch_hourly_weather_observations(latitude=12.9184, longitude=79.1325)
            # Must remain 1 because it read from cache
            self.assertEqual(mock_get.call_count, 1)
            self.assertEqual(res1["weather_df"].iloc[0]["weather_pressure"], res2["weather_df"].iloc[0]["weather_pressure"])

    def test_03_timestamp_alignment_interpolation_and_boundary(self):
        """Validates linear interpolation on 10-minute cadence and boundary rejection."""
        base_t = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
        weather_df = pd.DataFrame({
            "ts_utc": [base_t, base_t + timedelta(hours=1)],
            "weather_pressure": [1000.0, 1006.0],  # 0.1 hPa per minute
            "weather_wind_speed": [10.0, 16.0],
            "weather_radiation": [100.0, 220.0],
        })

        # Observations at 12:10, 12:20, 12:30
        obs_times = [
            base_t + timedelta(minutes=10),
            base_t + timedelta(minutes=20),
            base_t + timedelta(minutes=30),
        ]

        aligned = align_weather_to_observation_timestamps(obs_times, weather_df)
        self.assertIsNotNone(aligned)
        self.assertEqual(len(aligned), 3)
        # 12:10 should be 1000.0 + 10/60 * 6.0 = 1001.0
        self.assertAlmostEqual(aligned["weather_pressure"].iloc[0], 1001.0, places=4)
        # 12:30 should be 1000.0 + 30/60 * 6.0 = 1003.0
        self.assertAlmostEqual(aligned["weather_pressure"].iloc[2], 1003.0, places=4)

        # Boundary test: observation before weather start must be rejected (no extrapolation)
        bad_obs_times = [base_t - timedelta(minutes=10), base_t + timedelta(minutes=10)]
        self.assertIsNone(align_weather_to_observation_timestamps(bad_obs_times, weather_df))

    def test_04_api_failure_and_unconfigured_coordinates(self):
        """Validates graceful degradation when coordinates are missing or API fails."""
        # Unconfigured coordinates
        res_none = fetch_hourly_weather_observations(latitude=None, longitude=None)
        self.assertFalse(res_none["available"])
        self.assertIn("coordinates", res_none["error"].lower())

        # Network timeout
        with patch("requests.get", side_effect=Exception("Connection timeout")):
            res_fail = fetch_hourly_weather_observations(latitude=12.9184, longitude=79.1325)
        self.assertFalse(res_fail["available"])
        self.assertIn("timeout", res_fail["error"].lower())

    def test_05_strict_zero_fabrication_guard(self):
        """Confirms that incomplete live candidate lacking the 4 in-situ channels is strictly rejected."""
        # Sequence with 24 timesteps having edge + external weather, but missing 4 in-situ channels
        incomplete_payload = pd.DataFrame({
            "soil_moisture": [45.0] * 24,
            "weather_temp": [25.0] * 24,
            "weather_humidity": [60.0] * 24,
            "weather_rain": [0.0] * 24,
            "weather_pressure": [1013.0] * 24,
            "weather_wind_speed": [12.0] * 24,
            "weather_radiation": [200.0] * 24,
            "soil_temperature_0-7cm": [22.0] * 24,
            "hour": [12] * 24,
            "month": [10] * 24,
            # Notably MISSING: soil_temperature_7-18cm, ec, ph, irrigation_duration_minutes
        })

        lstm_model, lstm_meta = load_forecasting_model("LSTM")
        with self.assertRaises(ValueError) as ctx:
            preprocess_sequence(incomplete_payload, lstm_meta["scaler_mean"], lstm_meta["scaler_scale"])
        self.assertIn("missing required features", str(ctx.exception).lower())

    def test_06_safety_invariants(self):
        """Hard locks: PUMP_ALLOWED is False."""
        self.assertFalse(PUMP_ALLOWED)


if __name__ == "__main__":
    unittest.main()

# Phase 24 — Real-World Sensor Data Collection and Calibration Pipeline Report

**Project Root:** `/Users/aryanagrawal/ESA_PROJECT`  
**Audit Date:** October 10, 2026  
**Status:** Pipeline Implemented, Verified, and Documented. Zero model training.  
**Safety Locks:** `PUMP_ALLOWED = False` hard-locked. Phase 11 production classifier preserved.  

---

## Executive Summary

Phase 24 established the real-world sensor data collection, observation aggregation, and calibration pipeline for the Smart Agriculture System (`ESA_PROJECT`). It provides a rigorous bridge between raw ESP32 serial telemetry and future deep sequence forecasting models (LSTM, TCN, and Transformer) without modifying production irrigation logic or fabricating missing environmental channels.

### Key Outcomes:
1. **Raw Telemetry Preservation:** The existing SQLite schema (`sensor_readings` in `data/sensor_history.db`) remains intact and unmodified. All 6,700+ raw ADC counts and DHT22 readings are preserved. The physical distinction between high-frequency raw serial packets (~2.5s), host receipt timestamps, and aggregated 10-minute observation buckets is formally codified.
2. **Deterministic 10-Minute Observation Aggregation:** Implemented `aggregate_raw_telemetry()` in `training/experimental/phase24/data_collection.py`. It deterministically buckets raw readings into non-overlapping 600-second intervals aligned to `floor(epoch / 600) * 600`, calculating sample count, time coverage, means, standard deviations, and valid measurement counts. Empty intervals and disconnect gaps are explicitly identified without synthetic interpolation.
3. **Rain Sensor Differentiation:** Explicitly codified the physical distinction between the ESP32 digital raindrop sensor (which detects surface wetness ratio $0.0 - 1.0$) and rainfall depth ($mm$) required by the training models. Intervals with dry indications are strictly distinguished from intervals with missing or unread rain data.
4. **Water Level vs. Irrigation Volume:** Confirmed that `water_level_raw` measures reservoir height, **not** irrigation volume or valve actuation duration. Actuator code is disabled (`PUMP_ALLOWED = False`).
5. **Soil Moisture Calibration Data Standard:** Designed `validate_calibration_record()`. Specifies mandatory fields (`probe_id`, `raw_adc`, `timestamp`, `sample_id`, `reference_value`, `reference_unit`) and strictly rejects ungrounded, fabricated, or unit-mismatched calibration records.
6. **Feature Readiness Audit:** All 14 features required by the sequence models were audited:
   - 4 Available from genuine live data (`weather_temp`, `weather_humidity`, `hour`, `month`).
   - 2 Available but incompatible in scale or meaning (`weather_rain`, `irrigation_duration_minutes`).
   - 1 Requires physical soil core calibration (`soil_moisture`).
   - 7 Completely missing physical channels on edge hardware (`weather_pressure`, `weather_wind_speed`, `weather_radiation`, `soil_temperature_0-7cm`, `soil_temperature_7-18cm`, `ec`, `ph`).
7. **Dashboard Integration:** Enhanced the `'🤖 AI Models'` tab in `dashboard.py` with a live telemetry coverage audit, probe calibration status cards, a 14-feature readiness table, and the mandatory warning:
   > **`LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL`**
8. **Test Verification:** Implemented 10 tests in `test_phase24_data_collection.py`. Ran all 6 test suites with 72 out of 72 tests passing in 9.143 seconds.

---

## 1. Raw Telemetry Preservation & Pipeline Distinction

The physical and architectural boundaries between telemetry stages are defined as follows:

```
+-----------------------------------------------------------------------------------+
| 1. RAW SERIAL PACKETS                                                            |
| Source: ESP32 loop (delay 2500ms). Frequency: ~0.4 Hz (every ~2.517 seconds).    |
| Content: Uncalibrated ADC counts (soil1_raw, soil2_raw, water_level_raw, rain_pin) |
|          DHT22 air readings (temperature_c, humidity_pct), rain_detected boolean. |
| Limitation: ESP32 lacks RTC; contains NO device timestamp.                       |
+-----------------------------------------------------------------------------------+
                                        │
                                        ▼
+-----------------------------------------------------------------------------------+
| 2. HOST RECEIPT TIMESTAMPS & PERSISTENT SQLITE STORAGE                            |
| Host: Mac Python thread in dashboard.py / esp32_reader.py.                        |
| Table: sensor_readings in data/sensor_history.db.                                |
| Timestamp: datetime.now(timezone.utc).isoformat() & time.time() epoch float.      |
| Integrity: INSERT OR IGNORE, zero modification of existing rows.                  |
| Limitation: Represents Mac clock arrival time, not sensor sampling instant.       |
+-----------------------------------------------------------------------------------+
                                        │
                                        ▼
+-----------------------------------------------------------------------------------+
| 3. AGGREGATED TEN-MINUTE OBSERVATIONS (Phase 24)                                  |
| Boundary: floor(epoch / 600.0) * 600.0 (UTC aligned).                            |
| Nominal span: Exactly 600 seconds (10 minutes).                                   |
| Aggregation: sample_count, time_coverage, mean, std, valid counts.                |
| Integrity: Non-destructive, deterministic, no synthetic filling of empty buckets. |
+-----------------------------------------------------------------------------------+
```

### Migration and Compatibility Confirmation
- No changes were made to existing tables `sensor_readings` or `probe_calibration`.
- Existing SQLite queries in `database.py` and `dashboard.py` remain 100% compatible.
- All 6,700+ raw records recorded since system commissioning remain untouched.

---

## 2. Deterministic Ten-Minute Observation Aggregation

Implemented in `training/experimental/phase24/data_collection.py`:
`aggregate_raw_telemetry(df_raw, interval_seconds=600.0)`

### Boundary & Timezone Rules:
- **Bucket Start Epoch:** $t_{\text{start}} = \lfloor \frac{\text{recorded\_at}}{600} \rfloor \times 600$.
- **Bucket End Epoch:** $t_{\text{end}} = t_{\text{start}} + 600$.
- **Timezone:** Standardized UTC ISO 8601 formatting (`datetime.fromtimestamp(epoch, tz=timezone.utc)`).

### Aggregation Fields:
1. `raw_sample_count`: Total valid packets logged within the 10-minute window (nominally ~238 packets at 2.5s cadence).
2. `time_coverage_seconds`: Elapsed seconds between the first and last recorded packet in the window ($t_{\text{last}} - t_{\text{first}}$).
3. `soil1_raw_mean` & `soil1_raw_std`: Mean and standard deviation of capacitive Probe 1 ADC.
4. `soil2_raw_mean` & `soil2_raw_std`: Mean and standard deviation of capacitive Probe 2 ADC.
5. `temperature_c_mean` & `temperature_c_valid_count`: Mean DHT22 air temperature and count of non-null readings.
6. `humidity_pct_mean` & `humidity_pct_valid_count`: Mean DHT22 air humidity and count of non-null readings.
7. `rain_detected_count`: Number of readings in bucket with active rain indication (`rain_detected == 1`).
8. `has_valid_rain_data`: Boolean flag indicating whether non-null rain readings exist.
9. `rain_presence_ratio`: Float ratio in $[0.0, 1.0]$ representing the fraction of readings with rain detected (or `None` if no rain data was collected).
10. `water_level_raw_mean`: Mean raw ADC of the reservoir water depth probe.

### Rain Sensor Distinction:
- The ESP32 digital raindrop board (GPIO 27) operates via surface water bridging. It produces a binary boolean: `rain_detected = (rain_pin == LOW)`.
- It registers surface moisture presence, **not** accumulated precipitation depth in millimeters ($mm$).
- In contrast, the training dataset's `weather_rain` feature represents rainfall depth in $mm$ measured by an ERA5 tipping bucket gauge.

### Water Level Distinction:
- `water_level_raw` (GPIO 34) measures the hydrostatic depth of water in an external reservoir.
- It does **not** measure irrigation volume delivered to the root zone, nor does it represent valve actuation duration.
- Pump actuation is hard-locked (`PUMP_ALLOWED = False`).

---

## 3. Soil Probe Calibration Data Format & Validation

Implemented in `training/experimental/phase24/data_collection.py`:
`validate_calibration_record(record)`

### Calibration Record Schema:

| Field Name | Type | Mandatory? | Validation Rule |
| :--- | :---: | :---: | :--- |
| `probe_id` | `str` | **Yes** | Must be `'probe1'` or `'probe2'`. |
| `raw_adc` | `float` | **Yes** | Numeric value within 12-bit range $[0.0, 4095.0]$. |
| `timestamp` | `float` or `str` | **Yes** | Unix epoch float or valid ISO 8601 string. |
| `sample_id` | `str` | **Yes** | Non-empty string identifier (e.g. `'CORE_ZONE1_01'`). |
| `reference_value` | `float` | **Yes** | Laboratory reference measurement. **Never generated automatically.** |
| `reference_unit` | `str` | **Yes** | Must be one of `['vwc_pct', 'vwc_ratio', 'gravimetric_pct']`. |
| `soil_type` | `str` | No | Text description (e.g. `'Sandy Loam'`, `'Clay'`). |
| `bulk_density_g_cm3` | `float` | No | Dry soil bulk density ($\rho_b > 0$). |
| `sample_temperature_c` | `float` | No | Temperature of soil sample during measurement (°C). |
| `calibration_method` | `str` | No | Procedure description (e.g. `'gravimetric_oven_dry'`). |
| `notes` | `str` | No | Experimental observations or operator notes. |

### Strict Enforcement:
- Rejects any calibration record missing ground-truth reference values.
- Rejects unit mismatches (e.g. values exceeding $1.0$ for `vwc_ratio` or $100.0$ for `vwc_pct`).
- Does not modify existing production provisional constants (Probe 1: 2054 dry / 1073 wet; Probe 2: 1906 dry / 1023 wet).

---

## 4. 14-Feature Input Readiness Breakdown

The 14 features expected by the trained LSTM, TCN, and Transformer models were audited:

| # | Feature Name | Trained Units & Meaning | Live ESP32 Hardware Source | Category Status | Specific Blocker / Architectural Gap |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | `weather_temp` | Ambient air temp (°C), ERA5 | DHT22 on GPIO 4 | Available from genuine live data | Scale compatible. Requires contiguous 4h 10m buffer. |
| **2** | `weather_humidity` | Relative humidity (%), ERA5 | DHT22 on GPIO 4 | Available from genuine live data | Scale compatible. Requires contiguous 4h 10m buffer. |
| **3** | `weather_rain` | Rainfall depth (mm/10min), ERA5 | Rain board on GPIO 27 | Available but incompatible in scale or meaning | Qualitative surface water presence (0/1), not depth in mm. |
| **4** | `weather_pressure` | Barometric pressure (hPa), ERA5 | None installed | Missing | No barometric pressure sensor on edge station. |
| **5** | `weather_wind_speed` | Wind speed (m/s), ERA5 | None installed | Missing | No anemometer on edge station. |
| **6** | `weather_radiation` | Solar radiation flux (W/m²), ERA5 | None installed | Missing | No pyranometer on edge station. |
| **7** | `soil_temperature_0-7cm` | Topsoil temp 0–7 cm (°C) | None installed | Missing | No buried topsoil thermistor. DHT22 measures air temp. |
| **8** | `soil_temperature_7-18cm` | Subsoil temp 7–18 cm (°C) | None installed | Missing | No buried subsoil thermistor. |
| **9** | `ec` | Soil electrical conductivity (mS/cm) | None installed | Missing | No soil EC probe installed. |
| **10** | `ph` | Soil pH (0–14 scale) | None installed | Missing | No soil pH probe installed. |
| **11** | `soil_moisture` | Volumetric water content (%) | Capacitive Probes (GPIO 32/33) | Requires calibration | Probes output air/water relative ADC index, not VWC. Requires gravimetric soil core calibration curve. |
| **12** | `irrigation_duration_minutes` | Valve actuation duration (min) | None / `water_level_raw` | Available but incompatible in scale or meaning | Water level sensor measures reservoir depth, not valve flow or duration. Pump is locked (`PUMP_ALLOWED=False`). |
| **13** | `hour` | Hour of day (0–23) | Host UTC clock | Available from genuine live data | Deterministic from sequence timestamp. |
| **14** | `month` | Month of year (1–12) | Host UTC clock | Available from genuine live data | Deterministic from sequence timestamp. |

### Quantitative Feature Summary:
- **Available from genuine live data:** 4 / 14 features
- **Requires calibration:** 1 / 14 features
- **Available but incompatible in scale or meaning:** 2 / 14 features
- **Missing physical channels:** 7 / 14 features

---

## 5. Dashboard Integration

In [`dashboard.py`](file:///Users/aryanagrawal/ESA_PROJECT/dashboard.py), Section 4 of the `'🤖 AI Models'` tab was enhanced to display the Phase 24 Data Readiness audit:
1. **Telemetry Coverage Cards:** Displays total raw samples (6,700+), aggregated 10-minute intervals (48+), disconnect gaps count, and total logging span (21+ hours).
2. **Soil Probe Calibration Status Cards:** Displays calibration status, dry/wet ADC counts, and notes clarifying that values represent provisional air/water indices, not physical VWC.
3. **14-Feature Readiness Table:** Full interactive dataframe rendering the exact status, source, units, scale compatibility, and blockers for each model feature.
4. **Mandatory Status Alert:**
   ```html
   🛑 LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL
   Readiness Summary: 4 / 14 features available • 1 requiring soil core calibration • 
   2 incompatible scale/meaning • 7 completely missing physical channels.
   Experimental live forecasting remains strictly disabled until physical weather/soil telemetry, 
   gravimetric soil calibration, and contiguous 4-hour historical buffers are implemented. 
   Missing sensor measurements are strictly never fabricated.
   ```

---

## 6. Test Suite & Verification Results

The test suite was extended with [`test_phase24_data_collection.py`](file:///Users/aryanagrawal/ESA_PROJECT/test_phase24_data_collection.py) containing 10 unit and integration tests.

### Complete Regression Run (All 6 Suites):
```bash
python3 -m unittest test_ml_integration.py test_phase14_integration.py test_phase15_transformer.py test_phase22_forecasting_integration.py test_phase23_sequence_buffer.py test_phase24_data_collection.py
```

```
----------------------------------------------------------------------
Suite                                     Tests Run   Passed   Failed
----------------------------------------------------------------------
test_ml_integration.py                       25         25        0
test_phase14_integration.py                  10         10        0
test_phase15_transformer.py                   10         10        0
test_phase22_forecasting_integration.py       7          7        0
test_phase23_sequence_buffer.py              10         10        0
test_phase24_data_collection.py               10         10        0
----------------------------------------------------------------------
Total Tests Executed                         72         72        0
Runtime: 9.143 seconds
Result: ALL 72 TESTS PASSED (100% SUCCESS, 0 REGRESSIONS)
```

---

## 7. Deliverables & Files Changed

1. **Created:** [`training/experimental/phase24/data_collection.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase24/data_collection.py)
   - Implements `aggregate_raw_telemetry()`, `identify_aggregation_gaps()`, `validate_calibration_record()`, `get_14_feature_readiness_audit()`, and `audit_sqlite_data_collection()`.
2. **Created:** [`test_phase24_data_collection.py`](file:///Users/aryanagrawal/ESA_PROJECT/test_phase24_data_collection.py)
   - 10 unit and integration tests verifying interval math, missing DHT22 handling, rain sensor differentiation, calibration record validation, and production safety locks.
3. **Created:** [`docs/phase24_sensor_data_collection.md`](file:///Users/aryanagrawal/ESA_PROJECT/docs/phase24_sensor_data_collection.md)
   - Full technical audit report.
4. **Modified:** [`dashboard.py`](file:///Users/aryanagrawal/ESA_PROJECT/dashboard.py)
   - Added Phase 24 Live Sensor Data Readiness and Calibration Pipeline section to `'🤖 AI Models'` tab without affecting production UI or inference.
5. **Preserved Intact:**
   - `training/models/random_forest_11feat.pkl` (unmodified)
   - `training/models/random_forest_11feat_metrics.json` (unmodified)
   - `esp32/sensor_test/sensor_test.ino` (unmodified)
   - `data/sensor_history.db` (unmodified)
   - `training/inference.py` (unmodified)
   - Hard safety locks: `PUMP_ALLOWED = False` preserved across all files.

# Phase 25 — Resolve Live Forecasting Blockers Report

**Project Root:** `/Users/aryanagrawal/ESA_PROJECT`  
**Audit Date:** October 10, 2026  
**Execution Type:** Technical blocker resolution, real-time history infrastructure implementation, dataset compatibility audit, and safety verification.  
**Operating Constraints:** Zero model retraining. Zero synthetic feature fabrication. `PUMP_ALLOWED = False` hard-locked.  

---

## Executive Summary

Phase 25 directly resolved the engineering blocker identified in prior phases regarding real-time history collection by implementing and integrating a dedicated, persistent 10-minute observation infrastructure directly into SQLite and the dashboard. It also conducted a rigorous scientific feasibility study on whether a hardware-compatible model can be trained on the existing dataset.

### Key Outcomes:
1. **Resolved Blocker: Missing Real-Time 10-Minute History Infrastructure:**
   - Implemented persistent table `ten_minute_observations` in `data/sensor_history.db`.
   - Developed `sync_raw_readings_to_ten_minute_observations()` and `extract_realtime_24step_hardware_sequence()`.
   - Verified that the system successfully synchronized 6,900+ raw serial records into 49 non-overlapping 10-minute buckets.
   - Discovered and extracted an actual **24-step (4.0-hour) continuous sequence of genuine ESP32 telemetry** spanning from `2026-10-10 13:00:00 UTC` to `16:50:00 UTC`.
2. **Resolved Blocker: Missing Calibration Observation Logger:**
   - Implemented persistent table `soil_calibration_logs` in SQLite.
   - Created `log_soil_calibration_measurement()` with strict validation rules requiring ground-truth laboratory reference values (`vwc_pct`, `vwc_ratio`, `gravimetric_pct`), sample IDs, and physical soil core properties.
3. **Training Data Compatibility Audit: Unresolvable with Current Datasets:**
   - The Arnesano precision-irrigation dataset (`Soil Moisture, Irrigation Actuator and Weather Dat`) cannot be retrofitted into a hardware-compatible model for our ESP32. It was collected on Apulia soil using research-grade FDR probes, ERA5 solar pyranometers, barometers, anemometers, and Modbus EC/pH sensors.
   - Training an experimental neural network on our 49 local 10-minute buckets (covering only ~21.7 hours of static bench testing) would result in extreme overfitting on a single quiescent regime with zero irrigation events and zero rainfall.
   - Consequently, **no model was trained in this phase**, preserving scientific integrity.
4. **Dashboard Integration:**
   - Enhanced the `'🤖 AI Models'` tab in `dashboard.py` to display the Real-Time Hardware History Infrastructure card (Phase 25), reporting synchronized buckets, continuous streak length, active sequence window boundaries, and live telemetry statistics.
5. **Regression Verification:**
   - All 79 tests across all 7 test suites passed in 9.176 seconds with 100% success.
   - Production 11-feature model weights, threshold (`0.3270985223048995`), and `PUMP_ALLOWED = False` remain strictly intact.

---

## 1. Inspection of Models, Preprocessing, and Training Dataset

### Existing Model Requirements
- **Models:** `lstm_experimental.pt` (6,689 params), `tcn_experimental.pt` (17,921 params), `transformer_experimental.pt` (18,113 params).
- **Mandatory Input Matrix:** Exactly 24 timesteps $\times$ 14 features = Tensor shape `(1, 24, 14)`.
- **Target Variable:** `target_mean_24h` (continuous mean volumetric soil moisture % over subsequent 24 hours, i.e., 144 steps @ 10m).
- **Embedded Scalers:** Checkpoints store fixed float32 vectors `scaler_mean` and `scaler_scale` of length 14. These weights cannot accept fewer features without retraining.

### Dataset Incompatibility Analysis
The Arnesano dataset cannot be used to train a reduced ESP32 model:
1. **Physical Sensor Mismatch:** The Arnesano dataset measures volumetric soil water retention using calibrated Campbell Scientific TDR/FDR probes. Our ESP32 uses low-cost capacitive probes measuring uncalibrated dielectric frequency/impedance counts.
2. **Missing Essential Atmospheric Drivers:** Soil moisture dynamics over 24 hours are governed by evapotranspiration ($ET_0$), driven primarily by solar radiation ($R_s$), vapor pressure deficit (derived from barometric pressure and temperature), and wind speed ($u_2$). None of these three physical drivers exist on the ESP32.
3. **No Cross-Device Transferability:** Transferring neural network weights trained on Italian Mediterranean silt loam to an uncalibrated indoor capacitive probe is physically ungrounded.

---

## 2. Hardware-Compatible Model Feasibility Study

An investigation was conducted on whether an experimental model could be trained using only genuinely available ESP32 hardware telemetry:
- `soil1_raw` / `probe1` relative moisture %
- `soil2_raw` / `probe2` relative moisture %
- `temperature_c` (DHT22 air temperature)
- `humidity_pct` (DHT22 air humidity)
- `rain_detected` (binary surface wetness indicator)
- `hour` and `month` (cyclical timestamp features)

### Findings:
1. **Lack of Labeled Training Dataset:** 
   Our SQLite database (`data/sensor_history.db`) currently contains 6,949 raw readings grouped into 49 ten-minute buckets spanning ~21.7 hours.
2. **Insufficient Temporal Horizon for 24-Hour Forecasting:**
   To calculate `target_mean_24h` for even a single sequence window, the system requires 24 hours of future data following the 4-hour input sequence (a continuous span of 28 hours minimum). The existing dataset span is only 21.7 hours total, broken by multi-hour gaps. Thus, **zero valid target labels exist** in the recorded telemetry.
3. **Scientific Conclusion:**
   Attempting to train an LSTM, TCN, or Transformer on 49 uncalibrated bench readings without ground-truth future labels would constitute mathematical fabrication. Training is deferred until a multi-week continuous telemetry campaign is completed.

---

## 3. Real-Time Hardware History Infrastructure (Task 4)

In Phase 25, the real-time history infrastructure was implemented in [`training/experimental/phase25/realtime_history.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase25/realtime_history.py):

### 3.1 Persistent Tables Initialized
- `ten_minute_observations`:
  - `interval_start_epoch` (REAL PRIMARY KEY, floor of epoch / 600)
  - `interval_end_epoch` (REAL)
  - `interval_start_utc` & `interval_end_utc` (ISO 8601 strings)
  - `raw_sample_count` (int, sample size in bucket)
  - `time_coverage_seconds` (float, span of readings in bucket)
  - `soil1_raw_mean`, `soil1_raw_std`, `soil2_raw_mean`, `soil2_raw_std` (float)
  - `temperature_c_mean`, `temperature_c_valid_count`
  - `humidity_pct_mean`, `humidity_pct_valid_count`
  - `rain_detected_count`, `rain_pin_raw_mean`, `has_valid_rain_data`, `rain_presence_ratio`
  - `water_level_raw_mean`, `status`, `created_at`
- `soil_calibration_logs`:
  - `id` (INTEGER PRIMARY KEY)
  - `probe_id`, `raw_adc`, `recorded_at`, `timestamp_utc`, `sample_id`
  - `reference_value`, `reference_unit`, `soil_type`, `bulk_density_g_cm3`
  - `sample_temperature_c`, `calibration_method`, `notes`, `created_at`

### 3.2 Synchronization & Sequence Extraction Results
- Synchronized raw records: **6,949 raw samples** $\to$ **49 distinct 10-minute buckets**.
- Contiguous streak evaluation:
  - Discovered a continuous streak of **24 consecutive 10-minute observations (4.0 hours continuous)** from `2026-10-10 13:00:00 UTC` to `2026-10-10 16:50:00 UTC`.
  - Mean Probe 1 ADC: $1065.7$
  - Mean Probe 2 ADC: $1030.5$
  - Mean Air Temp: $20.7^\circ\text{C}$
  - Mean Air Humidity: $69.2\%$

This confirms that the 4-hour temporal buffer blocker has been solved at the infrastructure level.

---

## 4. Calibration Observation Repository Implementation

Implemented `log_soil_calibration_measurement()` in `training/experimental/phase25/realtime_history.py`.

### Validation Rules Enforced:
1. `probe_id` must be `'probe1'` or `'probe2'`.
2. `raw_adc` must be a valid float in $[0.0, 4095.0]$.
3. `sample_id` must be a non-empty laboratory identifier.
4. `reference_value` must be a genuine measured number; automatic fabrication is strictly prohibited.
5. `reference_unit` must be one of `['vwc_pct', 'vwc_ratio', 'gravimetric_pct']`.

---

## 5. Dashboard Integration & User Interface

The `'🤖 AI Models'` tab in [`dashboard.py`](file:///Users/aryanagrawal/ESA_PROJECT/dashboard.py) was enhanced to display:
- **Synchronized 10m Buckets Metric:** Reflects live row count in `ten_minute_observations`.
- **Longest Continuous Streak Metric:** Shows continuous 10m steps and hours.
- **Hardware Buffer Status Banner:** Indicates green when $\ge 24$ continuous steps are available.
- **Sequence Inspection Tile:** Reports active start/end timestamps, mean raw ADC for both probes, and air temperature.
- **Persistent Disclaimer:** Live sequence inference remains blocked pending physical sensor additions and soil gravimetric calibration.

---

## 6. Test Suite Execution & Verification

Added 7 new unit and integration tests in [`test_phase25_realtime_history.py`](file:///Users/aryanagrawal/ESA_PROJECT/test_phase25_realtime_history.py).

### Complete Regression Run Across All 7 Suites:
```bash
python3 -m unittest test_ml_integration.py test_phase14_integration.py test_phase15_transformer.py test_phase22_forecasting_integration.py test_phase23_sequence_buffer.py test_phase24_data_collection.py test_phase25_realtime_history.py
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
test_phase25_realtime_history.py              7          7        0
----------------------------------------------------------------------
Total Tests Executed                         79         79        0
Runtime: 9.176 seconds
Result: ALL 79 TESTS PASSED (100% SUCCESS, ZERO REGRESSIONS)
```

---

## 7. Concrete Answers to Final Deliverable Requirements

### 1. Which blockers were actually resolved?
- **Blocker 1 (Missing 10-Minute Observation Infrastructure):** **RESOLVED.** Created persistent `ten_minute_observations` table in SQLite, synchronization logic, and sequence extractor. Successfully extracted real 24-step 4.0-hour hardware window from live telemetry.
- **Blocker 2 (Missing Calibration Data Repository):** **RESOLVED.** Created persistent `soil_calibration_logs` table and validation API to store empirical laboratory calibration measurements.
- **Blocker 3 (Real-time Telemetry Dashboard Visibility):** **RESOLVED.** Embedded real-time hardware history status cards directly into the Streamlit dashboard.

### 2. Whether compatible training data exists?
**NO.** The Arnesano dataset requires 8 physical sensors absent from the ESP32 and uses calibrated Italian agricultural soil. Our local SQLite database contains only ~21.7 hours of static bench testing, which lacks ground-truth future target labels (`target_mean_24h` requires 28+ hours of continuous history).

### 3. Whether a hardware-compatible model can be trained responsibly?
**NO, not yet.** Training a model on 49 static bench-test buckets without irrigation events or future labels would be scientifically invalid and overfit to noise. Training must be conducted after a multi-week data collection campaign.

### 4. What physical measurements still need to be collected?
1. **Soil Core Gravimetric Samples:** Oven-dry soil core measurements at known moisture levels to fit empirical calibration polynomials $VWC = f(\text{ADC})$ for Probe 1 and Probe 2.
2. **Live Atmospheric Weather Telemetry:** Solar radiation ($W/m^2$), wind speed ($m/s$), and barometric pressure ($hPa$) via a local weather station or real-time Open-Meteo API bridge.
3. **Continuous Field Telemetry:** Minimum 3–4 weeks of uninterrupted 10-minute telemetry during active plant growth with recorded irrigation actuation events.

### 5. Tests executed and results:
79 out of 79 tests passed across all 7 test suites in 9.176 seconds.

### 6. Remaining blockers and the next concrete action:
- **Remaining Blockers:** Missing physical weather sensors (radiation, wind, pressure), uncalibrated capacitive soil ADC (needs conversion to VWC), and insufficient historical duration for training labels.
- **Next Concrete Action:** Connect the Open-Meteo live weather API to the 10-minute observation pipeline to automatically record ambient solar radiation, barometric pressure, and wind speed alongside ESP32 telemetry, and perform laboratory gravimetric calibration on both soil probes.

---

## 8. Confirmation of System Preservation & Safety Locks

- **Zero Model Training / Retraining:** No models were trained or fine-tuned.
- **Production Classifier Preservation:** `training/models/random_forest_11feat.pkl` (threshold `0.3270985223048995`, 11 features) remains the active decision model.
- **Pump Safety Hard Lock:** `PUMP_ALLOWED = False` remains strictly enforced across all files. Zero actuator logic exists.

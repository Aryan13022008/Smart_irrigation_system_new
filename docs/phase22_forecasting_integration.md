# Phase 22 — Experimental Forecasting Integration & Verification Audit Report

**Project:** Smart Agriculture System (`ESA_PROJECT`)  
**Audit Date:** October 10, 2026  
**Scope:** Integration, Verification, and Presentation of Experimental 24-Hour Soil-Moisture Forecasting Models (LSTM, TCN, Transformer) into the Streamlit Dashboard.  
**Operating State:** Software integration and presentation only. **Zero model retraining.**  

---

## Executive Summary

Phase 22 successfully verified and enhanced the integration of all three trained deep sequence models (**Experimental LSTM**, **Experimental TCN**, and **Experimental Transformer**) into the Smart Agriculture Streamlit dashboard (`dashboard.py`). The models operate strictly within an isolated, research-only section labelled:
> **`⚠️ EXPERIMENTAL — NOT VALIDATED FOR LIVE FIELD CONTROL`**

Key operational and safety outcomes confirmed in Phase 22:
1. **Production Classifier Invariance:** The Phase 11 production 11-feature model (`training/models/random_forest_11feat.pkl`, tuned threshold `0.3270985223048995`) remains the sole active decision model for production irrigation advice.
2. **Pump Actuation Hard Lock:** Automatic pump actuation remains hard-locked to disabled (`PUMP_ALLOWED = False`) across all modules.
3. **Honest Sensor Mismatch Handling:** Live ESP32 telemetry (single-depth capacitive relative ADC probe + DHT22) lacks 8 of the 14 mandatory physical variables (EC, pH, solar radiation, pressure, wind speed, dual-depth soil temperature). Live hardware forecasting is explicitly marked as **`BLOCKED / NOT VALIDATED`** with zero feature fabrication.
4. **Side-by-Side Verification:** The dashboard features side-by-side comparative architecture cards, chronological benchmark metrics against the persistence baseline, and an offline interactive demonstration running real-time PyTorch inference over stored 24-step reference sequences from the multi-sector research dataset.
5. **Comprehensive Verification:** 52 out of 52 unit and integration tests passed across all 4 test suites in ~8.5s.

---

## 1. Inspection and Integration Audit

Prior to making any edits, the existing codebase was thoroughly audited:
- `dashboard.py`: Contains 6 primary navigation pages (`🏠 Overview`, `💧 Irrigation`, `🌿 Weed Detection`, `🌦 Weather`, `🧪 Recommendations`, `🤖 AI Models`). The AI Models tab housed the Phase 11 11-feature audit and a preliminary benchmark summary table from Phase 14.
- `training/experimental/phase13/`: Contains `forecasting_models.py`, `lstm_experimental.pt`, `tcn_experimental.pt`, `phase13_lstm_tcn_results.json`, and `phase13_lstm_tcn_results.md`.
- `training/experimental/phase15_transformer/`: Contains `transformer_inference.py`, `transformer_experimental.pt`, `phase15_transformer_results.json`, and `phase15_transformer_results.md`.
- `training/inference.py`: Provides unified production inference and readiness audits. Kept completely intact and isolated from experimental forecasting.
- `Soil Moisture, Irrigation Actuator and Weather Dat/02_processed_data/preprocessed/`: Contains the verified 14-feature preprocessed dataset files for Sectors 1, 2, 3, 4, and 5.

**Design Decision:** Rather than introducing redundant inference files or polluting production inference modules, Phase 22 reused the clean, modular interfaces in `training/experimental/phase13/forecasting_models.py` (`load_forecasting_model`, `predict_future_moisture`) and `training/experimental/phase15_transformer/transformer_inference.py` (`load_transformer_model`, `predict_transformer_future_moisture`).

---

## 2. Experimental Model Loading and Checkpoints

All three model checkpoints were loaded, inspected, and verified in `torch.eval()` mode with zero parameter gradients enabled:

| Model Name | Checkpoint Path | Architecture Summary | Total Parameters | Sequence Length | Target Variable |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Experimental LSTM** | `training/experimental/phase13/lstm_experimental.pt` | 1-Layer LSTM (in=14, hidden=32) + MLP Head (32->16->1) | **6,689** | 24 steps (4h @ 10m) | `target_mean_24h` |
| **Experimental TCN** | `training/experimental/phase13/tcn_experimental.pt` | 3 Dilated Causal Residual Blocks (channels=[32,32,32], kernel=3, dilations=[1,2,4]) + Chomp1d + Linear Head | **17,921** | 24 steps (4h @ 10m) | `target_mean_24h` |
| **Experimental Transformer** | `training/experimental/phase15_transformer/transformer_experimental.pt` | Positional Encoding + 2-Layer TransformerEncoder (d_model=32, 4 heads, d_ff=64, dropout=0.2) + Linear Head (32->16->1) | **18,113** | 24 steps (4h @ 10m) | `target_mean_24h` |

### Feature Scaling Properties
All three models share identical feature scalers stored in their checkpoint dictionaries:
- `scaler_mean`: `shape (14,)`, float32 array computed strictly on the training set.
- `scaler_scale`: `shape (14,)`, float32 standard deviation array, strictly positive (`scale > 0`).
- Standardization equation: $x_{\text{norm}} = \frac{x - \mu}{\sigma}$.

---

## 3. Shared 14-Feature Input Schema

All sequence models require exactly the following 14 features in exact order:

```
Index  Feature Name                   Physical Units / Meaning
-----------------------------------------------------------------------------
 0     weather_temp                   Ambient air temperature (°C)
 1     weather_humidity               Relative air humidity (%)
 2     weather_rain                   Precipitation / rainfall (mm)
 3     weather_pressure               Atmospheric pressure (hPa)
 4     weather_wind_speed             Wind speed (m/s)
 5     weather_radiation              Global solar radiation (W/m²)
 6     soil_temperature_0-7cm         Topsoil temperature 0–7 cm depth (°C)
 7     soil_temperature_7-18cm        Subsoil temperature 7–18 cm depth (°C)
 8     ec                             Bulk electrical conductivity (mS/cm)
 9     ph                             Soil pH acidity/alkalinity
10     soil_moisture                  Calibrated volumetric water content (%)
11     irrigation_duration_minutes    Prior actuator duration (min)
12     hour                           Hour of day (0–23)
13     month                          Month of year (1–12)
```

**Target Formulation:**
- `target_mean_24h`: Mean calibrated soil moisture over the subsequent 24 hours (144 future steps @ 10-minute sampling).
- This is a continuous regression task, **not** an irrigation ON/OFF actuation classifier.

---

## 4. Sensor-Data Mismatch Analysis: Live Hardware vs. Experimental Models

A critical engineering outcome of Phase 22 is documenting the sensor mismatch between the live edge deployment and the deep forecasting research models:

| Sensor / Channel | Live ESP32 Hardware Station | 14-Feature Research Models | Status & Handling |
| :--- | :--- | :--- | :--- |
| **Soil Moisture** | Relative probe ADC (air=0%, water=100%) | Calibrated volumetric water retention (%) | **Decoupled Scale**: Uncalibrated ADC cannot represent volumetric water content. |
| **Air Temperature** | DHT22 Sensor (°C) | ERA5 Weather Station (°C) | Partial match. |
| **Air Humidity** | DHT22 Sensor (%) | ERA5 Weather Station (%) | Partial match. |
| **Rain / Precipitation** | Binary / analog rain raindrop detector | ERA5 precipitation (mm) | Qualitative mismatch. |
| **Atmospheric Pressure** | Not installed | ERA5 barometric pressure (hPa) | **Missing on Hardware** |
| **Wind Speed** | Not installed | ERA5 wind anemometer (m/s) | **Missing on Hardware** |
| **Solar Radiation** | Not installed | Pyranometer solar flux (W/m²) | **Missing on Hardware** |
| **Soil Temp (0–7 cm)** | Not installed | Buried thermistor (°C) | **Missing on Hardware** |
| **Soil Temp (7–18 cm)**| Not installed | Buried subsoil thermistor (°C) | **Missing on Hardware** |
| **Electrical Conductivity (EC)** | Not installed | TDR soil conductivity probe (mS/cm) | **Missing on Hardware** |
| **Soil pH** | Not installed | Calibrated pH electrode probe | **Missing on Hardware** |
| **Temporal Sequence** | Instantaneous single-point packet | 24 contiguous 10-minute records (4h history) | **Missing History**: ESP32 buffer does not store 4h sequence. |

### Integrity Policy
1. **Zero Fabrication:** Missing sensors (pressure, solar radiation, EC, pH, deep soil temp) are never silently filled with dataset medians or simulated constants for live inference.
2. **Hardware Blocking:** Live hardware forecasting is permanently marked as **`BLOCKED / NOT VALIDATED`**.
3. **Reference Verification Only:** Interactive model demonstration in the dashboard is strictly fed with genuine 24-step contiguous records from the stored multi-sector research dataset.

---

## 5. Dashboard Implementation Architecture

In `dashboard.py` (lines 2090–2250), the experimental forecasting section has been structured into four clear visual tiers:

1. **Mandatory Safety Disclaimer Banner:**
   Rendered in high-visibility warning container:
   `⚠️ EXPERIMENTAL — NOT VALIDATED FOR LIVE FIELD CONTROL`
   Explicitly notes that predictions are exploratory regressions and automatic pump actuation is hard-locked (`PUMP_ALLOWED=False`).
2. **Chronological Benchmark Table & Scientific Findings:**
   Displays the benchmark comparison of Persistence Baseline vs. LSTM vs. TCN vs. Transformer on $N = 6,348$ held-out test windows. Summarizes the superiority of persistence and Simpson's Paradox in pooled $R^2$.
3. **Four Side-by-Side Model Architecture Cards:**
   - **Persistence Baseline:** 0 parameters, $x[t] \to x[t+24\text{h}]$, Test MAE: 0.412%, Active Reference Benchmark.
   - **Experimental LSTM:** 6,689 parameters, LSTM(14->32), FC(32->16->1), Test MAE: 3.867%, Live Status: Blocked.
   - **Experimental TCN:** 17,921 parameters, Dilated Conv1D [32,32,32], Test MAE: 4.691%, Live Status: Blocked.
   - **Experimental Transformer:** 18,113 parameters, 2-Layer Encoder (4 heads, d_model=32), Test MAE: 6.124%, Live Status: Blocked.
4. **Offline Inference Demonstration on Stored Multi-Sector Data:**
   Allows the user to select stored evaluation sequences from Sector 1 (Kiwi), Sector 2 (Vineyard), Sector 4 (Zucchini), or Sector 5 (Blueberry). Real-time PyTorch inference executes and displays live metric tiles for:
   - Persistence Baseline
   - LSTM Forecast (with $\Delta$ vs. Persistence)
   - TCN Forecast (with $\Delta$ vs. Persistence)
   - Transformer Forecast (with $\Delta$ vs. Persistence)
   - Ground Truth Target (`target_mean_24h`)
5. **Live Hardware Warning Tile:**
   Explicitly informs operators that the ESP32 edge station lacks 8 environmental channels and live forecasting is blocked.

---

## 6. Verification & Test Results

The test suite was expanded with `test_phase22_forecasting_integration.py` containing 7 dedicated unit and integration tests.

### Test Suite Execution Summary

```
----------------------------------------------------------------------
Suite                                     Tests Run   Passed   Failed
----------------------------------------------------------------------
test_ml_integration.py                       25         25        0
test_phase14_integration.py                  10         10        0
test_phase15_transformer.py                   10         10        0
test_phase22_forecasting_integration.py       7          7        0
----------------------------------------------------------------------
Total                                        52         52        0
Runtime: 8.478 seconds
```

### Key Assertions Verified in `test_phase22_forecasting_integration.py`:
- `test_01_all_three_checkpoints_exist_and_load`: Verifies that `ExpLSTM`, `ExpTCN`, and `ExpTransformer` checkpoints load cleanly, have parameter counts matching 6,689, 17,921, and 18,113, and are in `eval()` mode.
- `test_02_shared_feature_schema_and_scaler_properties`: Confirms that all 3 models share the identical 14-feature list and have positive feature scaling standard deviations.
- `test_03_inference_on_stored_evaluation_sequence`: Feeds real 24-step sequence from `dataset_zone_1_preprocessed.csv` and verifies that all models produce finite scalar predictions within reasonable ranges.
- `test_04_strict_input_validation_and_rejection`: Asserts that sequences with invalid row counts (10 or 25 rows), missing feature columns (`ec`), and incorrect array dimensions immediately raise `ValueError`.
- `test_05_esp32_sensor_mismatch_and_hardware_blocking`: Simulates live ESP32 telemetry, confirms the absence of 8 weather/soil channels, and validates that single-point dictionaries are rejected with `TypeError`.
- `test_06_production_11_feature_model_and_safety_invariance`: Confirms `PUMP_ALLOWED = False`, `ACTIVE_MODEL_NAME = "Random Forest (11-Feature)"`, and threshold is locked to `0.3270985223048995`.
- `test_07_dashboard_headless_execution`: Runs Streamlit headless `AppTest` over all navigation views (`🏠 Overview`, `💧 Irrigation`, `🤖 AI Models`) with 0 exceptions.

---

## 7. Sample Inference Output on Stored Reference Data

Real-time PyTorch forward pass executed on the final 24-step sequence from Sector 1 (Kiwi Orchard, held-out test split):

```
Sequence Horizon: 24 steps @ 10-minute cadence (4.0 hours history)
Ground Truth Target (target_mean_24h): 19.19%

Model Forecasts:
-------------------------------------------------------------
Persistence Baseline:      19.00%   (Error: -0.19%)
Experimental LSTM:         18.87%   (Error: -0.32%, Delta vs Persist: -0.13%)
Experimental TCN:          20.74%   (Error: +1.55%, Delta vs Persist: +1.74%)
Experimental Transformer:  20.14%   (Error: +0.95%, Delta vs Persist: +1.14%)
-------------------------------------------------------------
```

**Observation:** Even on an individual test window, the simple persistence heuristic ($19.00\%$) tracks the future 24-hour mean ($19.19\%$) more closely than the deep models, aligning with the chronological test set findings where post-season soil moisture remains in near-static equilibrium.

---

## 8. Preserved System Features & Zero Regression Guarantee

During Phase 22, the following system features were strictly preserved without modification:
1. `training/models/random_forest_11feat.pkl`: Unmodified binary weights.
2. `training/models/random_forest_11feat_metrics.json`: Unmodified evaluation metrics.
3. Feature ordering in production inference: 11 exact features preserved.
4. Decision threshold: Locked at `0.3270985223048995`.
5. Firmware & Database: ESP32 firmware source code, SQLite database schema, and weed detection YOLO models were untouched.
6. Zero Model Retraining: No neural networks or tree ensembles were trained or retrained.

---

## 9. Operational & Scientific Recommendations

1. **Production Deployment Recommendation: NO-GO for Neural Pump Control**
   The neural forecasting models (LSTM, TCN, Transformer) must **not** be deployed for pump control or live decision making. The naive persistence baseline outperforms all neural models by a factor of 10 on the test split (MAE 0.412% vs. 3.867% / 4.691% / 6.124%).
2. **Telemetry Gap:** The live ESP32 field station cannot support 14-channel sequence forecasting without installing weather station hardware (solar radiation, wind speed, barometric pressure) and multi-depth specialized soil probes (EC, pH, dual-depth temperature).
3. **Future Research Direction:**
   - Investigate event-conditioned forecasting where sequence models only activate during active irrigation seasons (as analyzed in Phases 18 and 19).
   - Collect local station history on the ESP32 using an SD card or server-side ring buffer to build multi-hour sequences before considering on-device sequence inference.

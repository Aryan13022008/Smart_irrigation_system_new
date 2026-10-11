# Phase 23 — Live Sensor Data Readiness and Forecasting Bridge Audit Report

**Project Root:** `/Users/aryanagrawal/ESA_PROJECT`  
**Audit Date:** October 10, 2026  
**Status:** Audit & Safety Bridge Completed. Zero model training. Zero model retraining.  
**Safety Locks:** `PUMP_ALLOWED = False` hard-locked. Phase 11 production classifier preserved.  

---

## Executive Summary

Phase 23 conducted an end-to-end audit of the real ESP32-to-dashboard telemetry pipeline and evaluated the physical, temporal, and mathematical readiness of live hardware data for deep sequence forecasting models (**Experimental LSTM, TCN, and Transformer**). 

### Key Audit Findings:
1. **Telemetry Pipeline Architecture:** ESP32 firmware emits JSON telemetry via USB serial every 2.5 seconds. SQLite (`data/sensor_history.db`) records packets with host receipt timestamps (`recorded_at` epoch seconds and UTC ISO strings).
2. **Four-Hour History Buffer Feasibility:** Although SQLite contains over 6,500 genuine sensor readings spanning ~21.7 hours, **a valid four-hour, 24-step sequence at 10-minute intervals cannot be constructed**. The stored data consists of high-frequency bursts (median $\Delta t \approx 2.52$s) interspersed with multi-hour disconnect gaps (max gap $\approx 11.9$h). Treating 2.5s serial packets as 10-minute sequence steps is mathematically fraudulent and strictly rejected.
3. **Feature Gap:** The live ESP32 edge station provides only 3 physical measurements (capacitive soil probe ADC, air temperature, air humidity). It completely lacks **8 of the 14 mandatory channels** (Atmospheric Pressure, Wind Speed, Solar Radiation, Dual-depth Soil Temperatures, Electrical Conductivity, and Soil pH). Furthermore, the water level sensor measures reservoir depth, not irrigation actuator volume.
4. **Soil Moisture Calibration Decoupling:** The ESP32 capacitive probe relative index ($0\%$ in air, $100\%$ in water) represents an uncalibrated electrical impedance ratio, **not** the volumetric water retention percentage ($VWC$) of the training dataset. Converting raw ADC to true $VWC$ requires physical gravimetric soil core calibration.
5. **Operational Integration Decision:** **Option C** is the only scientifically defensible choice. Live forecasting must remain **BLOCKED** with the explicit system status:
   > **`LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL`**
6. **Safety & Verification:** An isolated validation module (`training/experimental/phase23/sequence_buffer.py`) and a comprehensive test suite (`test_phase23_sequence_buffer.py`) were implemented. All 62 tests across the complete test suite passed with 0 regressions. Production 11-feature model weights, tuned threshold (`0.3270985223048995`), and `PUMP_ALLOWED = False` remain untouched.

---

## Task 1 — Real Telemetry Pipeline Audit

### 1.1 Pipeline Architecture Flow

```
[ESP32 Microcontroller]
   │  Analog read: GPIO 32 (Soil 1), GPIO 33 (Soil 2), GPIO 34 (Water Level), GPIO 27 (Rain)
   │  Digital read: GPIO 4 (DHT22 Temperature & Humidity)
   │  Loop cadence: delay(2500) -> ~2.5 seconds
   ▼
[USB Serial Connection]
   │  Port: /dev/cu.usbserial-0001, Baud: 115200, Format: Plain JSON string + newline
   ▼
[esp32_reader.py / dashboard.py Serial Thread]
   │  Non-blocking background thread reading serial buffer with readline()
   │  JSON parse into Python dict
   ▼
[Host In-Memory State & SQLite Logger]
   │  In-memory: self.latest = payload, self.last_received_at = time.time()
   │  Persistent DB: database.log_reading(payload, source="esp32_usb", status="valid")
   ▼
[SQLite Storage: data/sensor_history.db]
   │  Table: sensor_readings
   │  Timestamps: Host UTC ISO string + recorded_at (host unix epoch float)
   ▼
[dashboard.py UI]
   │  Renders current readings, probe calibration UI, and production 11-feature inference
```

### 1.2 Telemetry Schema & Payload Fields

The ESP32 firmware (`esp32/sensor_test/sensor_test.ino`) emits the following JSON structure every 2.5 seconds:

| Emitted Field Name | Hardware Source / Pin | Raw Type | Value Range / Description |
| :--- | :--- | :---: | :--- |
| `soil1_raw` | Capacitive Soil Probe 1 (GPIO 32) | `int` | 0 – 4095 (12-bit ADC count) |
| `soil2_raw` | Capacitive Soil Probe 2 (GPIO 33) | `int` | 0 – 4095 (12-bit ADC count) |
| `water_level_raw` | Resistive Water Level Sensor (GPIO 34) | `int` | 0 – 4095 (12-bit ADC count, reservoir level) |
| `rain_pin_raw` | Digital Rain Board Sensor (GPIO 27) | `int` | 0 (wet / rain) or 1 (dry / no rain) |
| `rain_detected` | Evaluated logic (`rain == LOW`) | `bool` | `true` if rain detected, `false` otherwise |
| `temperature_c` | DHT22 Air Temperature (GPIO 4) | `float` | -40.0 to +80.0 °C (or `null` if read error) |
| `humidity_pct` | DHT22 Air Humidity (GPIO 4) | `float` | 0.0 to 100.0 % (or `null` if read error) |

### 1.3 Sampling Cadence, Timestamping, and Missing-Reading Behavior

1. **Measurement Time vs. Host Receipt Time:**
   The ESP32 firmware **has no real-time clock (RTC)** and emits no timestamp fields over serial.
   Consequently, the timestamps in `sensor_readings` (`timestamp` and `recorded_at`) represent **host receipt time** on the Mac when `database.log_reading()` is invoked.
2. **Sampling Frequency:**
   The ESP32 firmware loops with `delay(2500)`, producing a packet every ~2.517 seconds.
3. **Database Logging Behavior:**
   - Every valid serial packet received by the host is passed to `database.log_reading()`.
   - SQLite enforces `INSERT OR IGNORE` with unique constraints on `timestamp`.
   - In `data/sensor_history.db`:
     - **Total Records:** 6,500+ rows.
     - **Median Interval:** 2.517 seconds.
     - **Span:** 21.7 hours.
     - **Gaps:** 53 gaps exceed 10 seconds; 13 gaps exceed 10 minutes; the largest gap is **11.91 hours** (when the serial connection was closed or the board was unpowered).
4. **Missing-Reading Handling:**
   - If DHT22 fails to read, the ESP32 emits `"temperature_c": null` and `"humidity_pct": null`.
   - `database.log_reading()` safely parses `null` to Python `None` and records `NULL` in SQLite.
   - If serial disconnection occurs, the host thread logs error messages in memory without creating phantom database rows. **Zero synthetic rows are generated.**

---

## Task 2 — Four-Hour History Buffer Analysis & Implementation

### 2.1 Feasibility Assessment: Can Stored Telemetry Support 24 x 10m Sequence?

**Conclusion: NO.**
Although the database contains readings over a 21.7-hour span, a valid 4-hour sequence consisting of 24 observations at 10-minute intervals **cannot currently be formed**:
1. **Burst Logging:** Readings are recorded every 2.5 seconds while active, rather than at 10-minute intervals.
2. **Downsampling Gaps:** If downsampling is attempted by selecting every 240th packet ($240 \times 2.5\text{s} = 600\text{s} = 10\text{m}$), the multi-hour disconnect gaps break sequence continuity. The longest continuous logging session without disconnection in the database lasted only **2.08 hours** (2,970 consecutive 2.5s packets), which is insufficient for a 4.0-hour requirement.
3. **Anti-Cheating Rules:**
   - Treating consecutive 2.5-second packets as 10-minute sequence observations would compress a 1-minute test into a 4-hour window, violating temporal physics.
   - Duplicating the latest measurement across 24 steps would fabricate zero-variance synthetic history.
   - Interpolating over multi-hour disconnect gaps is unscientific.

### 2.2 Sequence Buffer Implementation (`training/experimental/phase23/sequence_buffer.py`)

A standalone validation and sequence-construction module was developed under `training/experimental/phase23/`:

- **Cadence Rules:** Requires exactly 24 observations spaced at $600 \pm 60$ seconds.
- **Probe Identity Lock:** Strictly checks that all observations originate from the same probe (`probe_id`).
- **Feature Schema Enforcement:** Requires all 14 physical channels.
- **Finite Value Verification:** Rejects `NaN`, `Inf`, or `null` entries.
- **Diagnostic Blocker Codes:**
  - `HIGH_FREQUENCY_PACKET_STREAM`: Triggered when minimum step $< 60$s (rejects 2.5s packet streams).
  - `INSUFFICIENT_OBSERVATIONS`: Triggered when $< 24$ rows exist.
  - `IRREGULAR_TEMPORAL_SPACING`: Triggered when time step is outside $[540\text{s}, 660\text{s}]$.
  - `NON_MONOTONIC_TIMESTAMPS`: Triggered by unordered or duplicate timestamps.
  - `MIXED_PROBE_IDENTITIES`: Triggered if Probe 1 and Probe 2 readings are blended.
  - `MISSING_FEATURE_COLUMNS`: Triggered when any of the 14 features are absent.
  - `NON_FINITE_FEATURE_VALUES`: Triggered if sensor readings contain `NaN` or `null`.

When executed against the actual `data/sensor_history.db`, the audit immediately returns:
```
Status: BLOCKED
Blocker Code: HIGH_FREQUENCY_PACKET_STREAM
Blocker Message: Detected high-frequency telemetry (minimum step = 2.5s). Raw 2.5-second serial packets cannot be treated as independent 10-minute sequence observations.
```

---

## Task 3 — Comprehensive 14-Feature Input Audit

The 14 features expected by the trained LSTM, TCN, and Transformer models were audited against live hardware and weather sources:

| # | Feature Name | Trained Units & Scale | Live Hardware Source | Live Available? | Scale Compatible? | Telemetry Gap & Work Required |
| :-: | :--- | :--- | :--- | :-: | :-: | :--- |
| **1** | `weather_temp` | Ambient air temp (°C), Open-Meteo ERA5 | DHT22 on GPIO 4 | **Yes** | **Yes** | Local probe temp matches ERA5 ambient temperature. |
| **2** | `weather_humidity` | Relative humidity (%), Open-Meteo ERA5 | DHT22 on GPIO 4 | **Yes** | **Yes** | DHT22 relative humidity matches ERA5 percentage. |
| **3** | `weather_rain` | Precipitation (mm/10min), ERA5 | Digital Rain Pin (GPIO 27) | **Partial** | **No** | Hardware provides binary wet/dry; models require quantitative precipitation depth in mm. |
| **4** | `weather_pressure` | Barometric pressure (hPa), ERA5 | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires barometric sensor (BMP280/BME280) or live weather API bridge. |
| **5** | `weather_wind_speed` | Wind speed (m/s), ERA5 | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires anemometer or live weather API bridge. |
| **6** | `weather_radiation` | Solar radiation flux (W/m²), ERA5 | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires pyranometer sensor or live solar API. |
| **7** | `soil_temperature_0-7cm` | Topsoil temp (°C), thermistor | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires buried DS18B20 or PT100 temperature probe at 0–7 cm depth. |
| **8** | `soil_temperature_7-18cm` | Subsoil temp (°C), thermistor | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires second buried temperature probe at 7–18 cm depth. |
| **9** | `ec` | Bulk electrical conductivity (mS/cm) | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires specialized soil EC probe (e.g. RS485 Modbus soil sensor). |
| **10** | `ph` | Soil pH (0–14 scale) | None (No sensor) | **No** | **No** | **Missing on hardware**. Requires calibrated glass/ISFET soil pH probe. |
| **11** | `soil_moisture` | Calibrated volumetric water content (%) | Capacitive Probe (GPIO 32/33) | **Partial** | **No** | **Decoupled Scale**. Probes output air/water relative ADC, not physical soil volumetric moisture. |
| **12** | `irrigation_duration_minutes` | Prior valve actuation duration (min) | None | **No** | **No** | `water_level_raw` measures reservoir height, **not** irrigation flow/duration. Pump is disabled (`PUMP_ALLOWED=False`). |
| **13** | `hour` | Hour of day (0–23) | Host system clock | **Yes** | **Yes** | Derived deterministically from sequence timestamp. |
| **14** | `month` | Month of year (1–12) | Host system clock | **Yes** | **Yes** | Derived deterministically from sequence timestamp. |

### Summary of Feature Readiness
- **Directly Available:** 4 / 14 features (`weather_temp`, `weather_humidity`, `hour`, `month`).
- **Incompatible Scale / Form:** 2 / 14 features (`weather_rain` binary vs mm, `soil_moisture` relative % vs volumetric %).
- **Completely Missing on Hardware:** 8 / 14 features (`weather_pressure`, `weather_wind_speed`, `weather_radiation`, `soil_temperature_0-7cm`, `soil_temperature_7-18cm`, `ec`, `ph`, `irrigation_duration_minutes`).
- **Feature Fabrication Forbidden:** In accordance with scientific data integrity standards, missing features must not be populated with training medians or fabricated constants.

---

## Task 4 — Soil-Moisture Calibration Readiness

### 4.1 Existing ADC Calibration Constants & Logic

Inspection of `database.py` and `dashboard.py` reveals the current calibration parameters:

- **12-bit ADC Raw Range:** $0 \le \text{ADC} \le 4095$.
- **Provisional Air/Dry and Water/Wet References:**
  - **Probe 1 (GPIO 32):**
    - $\text{Dry ADC (in air)} = 2054.0$
    - $\text{Wet ADC (submerged in water)} = 1073.0$
    - Spread: $\Delta \text{ADC} = 981$ counts.
  - **Probe 2 (GPIO 33):**
    - $\text{Dry ADC (in air)} = 1906.0$
    - $\text{Wet ADC (submerged in water)} = 1023.0$
    - Spread: $\Delta \text{ADC} = 883$ counts.
- **Current Relative Index Formula:**
  $$\text{Relative Moisture \%} = \text{clamp}\left(0.0, 100.0, \frac{\text{Dry ADC} - \text{Raw ADC}}{\text{Dry ADC} - \text{Wet ADC}} \times 100.0\right)$$

### 4.2 Why This Index Is Not Volumetric Water Content (VWC)

The existing formula is a **two-point electrical endpoint normalization**, not a volumetric water content measurement:
1. **Physical Meaning:** A reading of $50\%$ on this index merely indicates that the sensor's capacitive impedance is halfway between dry air and liquid water. It does **not** mean that $50\%$ of the soil volume is water.
2. **Soil Porosity & Field Capacity:** Real agricultural soils saturate at total pore volumes typically between $35\%$ and $50\%$ volumetric water content ($VWC$). Soil at field capacity is typically $20\% - 35\%$, and at permanent wilting point ($PWP$) is typically $8\% - 15\%$. An uncalibrated probe submerged in saturated soil might read anywhere from $40\%$ to $85\%$ on the relative scale depending on soil mineralogy and compaction.
3. **Dielectric Non-Linearity:** Capacitive soil moisture probes measure the apparent dielectric permittivity ($\varepsilon_r$) of the soil-water-air matrix. The relationship between permittivity and volumetric water content is non-linear (described by the Topp equation: $\theta = -0.053 + 0.0292\varepsilon - 5.5\times 10^{-4}\varepsilon^2 + 4.3\times 10^{-6}\varepsilon^3$). Linear interpolation between air and water completely misrepresents this curve.
4. **Temperature and Salinity Interference:** Changes in soil temperature and ion concentration (salinity/EC) shift capacitive probe ADC readings independently of water content. The training dataset used calibrated research-grade TDR/FDR sensors that internally temperature-compensated volumetric measurements.

### 4.3 Physical Reference Measurements Required for True VWC Calibration

To legitimately convert the ESP32 capacitive probe ADC readings into volumetric soil moisture compatible with the research models, the following physical calibration procedure is required:
1. **Soil-Specific Gravimetric Calibration:**
   - Collect core soil samples from the target field/container.
   - Insert the probe and measure raw ADC across varying known moisture states.
   - Oven-dry the soil cores at $105^\circ\text{C}$ for 24 hours to determine dry bulk density ($\rho_b$) and gravimetric water content ($\theta_g = \frac{m_{\text{wet}} - m_{\text{dry}}}{m_{\text{dry}}}$).
   - Convert to volumetric water content: $\theta_v = \theta_g \times \frac{\rho_b}{\rho_w}$.
2. **Empirical Curve Fitting:**
   - Fit a polynomial calibration curve relating raw ADC to true volumetric water content $\theta_v$:
     $$\theta_v = a \cdot \text{ADC}^2 + b \cdot \text{ADC} + c$$
3. **Temperature Compensation:**
   - Record temperature simultaneously to decouple dielectric temperature drift from actual moisture variations.

---

## Task 5 — Integration Outcome Evaluation

Three options were evaluated for integrating live ESP32 telemetry with the experimental sequence models:

| Option | Description | Scientific Feasibility | Safety & Risk Assessment | Recommended? |
| :---: | :--- | :---: | :--- | :---: |
| **A** | **Live forecasting directly from ESP32 data** | **Completely Unfeasible** | Severe violation of scientific integrity. Would require fabricating 8 missing channels and feeding uncalibrated relative ADC into models trained on calibrated VWC. High risk of generating nonsense predictions. | **REJECTED** |
| **B** | **Live forecasting after additional sensors, calibration, and buffer are available** | **Feasible in Future Phase** | Valid future engineering roadmap. Requires installing weather station telemetry, dual-depth soil thermistors, EC/pH sensors, soil gravimetric calibration, and a 10-minute cron logger. | **FUTURE GOAL** |
| **C** | **Continue offline forecasting with stored dataset sequences while improving live telemetry pipeline independently** | **Fully Feasible & Scientifically Defensible** | **Zero Risk**. Preserves production stability, respects model assumptions, prevents data fabrication, and allows methodical enhancement of edge telemetry without compromising system safety. | **SELECTED (RECOMMENDED)** |

### Defensible Operational Outcome:
**Option C is adopted.** Live forecasting on the dashboard is explicitly blocked with the clear status message:
> **`LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL`**
> *Blockers: 8 missing physical environmental channels, uncalibrated relative moisture index, and absent 10-minute history buffer.*

---

## Task 6 — Safety Verification & Test Execution

### 6.1 Test Suite Implementation (`test_phase23_sequence_buffer.py`)

A dedicated test suite was implemented containing 10 rigorous unit and integration tests:
1. `test_01_valid_sequence_passes`: Confirms valid 24-step 10-minute sequence is accepted.
2. `test_02_reject_fewer_than_24_observations`: Rejects candidate windows with $< 24$ rows.
3. `test_03_reject_irregular_spacing_and_gaps`: Rejects sequences with temporal gaps exceeding tolerance.
4. `test_04_reject_unordered_and_duplicate_timestamps`: Rejects non-monotonic and duplicate timestamps.
5. `test_05_reject_mixed_probe_identities`: Rejects sequences mixing Probe 1 and Probe 2 measurements.
6. `test_06_reject_missing_feature_columns`: Rejects incomplete feature subsets without fabrication.
7. `test_07_reject_non_finite_values`: Rejects `NaN`, `Inf`, or `null` values in feature channels.
8. `test_08_reject_high_frequency_packet_stream`: Rejects 2.5s serial loops masquerading as 10m steps.
9. `test_09_no_future_rows_and_trailing_window_alignment`: Confirms strictly trailing historical window alignment.
10. `test_10_production_invariance_and_safety_lock`: Asserts `PUMP_ALLOWED = False`, production model is 11-feature Random Forest, and threshold is `0.3270985223048995`.

### 6.2 Full Test Regression Results

```bash
python3 -m unittest test_ml_integration.py test_phase14_integration.py test_phase15_transformer.py test_phase22_forecasting_integration.py test_phase23_sequence_buffer.py
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
----------------------------------------------------------------------
Total Tests Executed                         62         62        0
Runtime: 8.211 seconds
Result: ALL 62 TESTS PASSED (100% SUCCESS)
```

---

## Answers to User Prompt Audit Questions

1. **What the actual telemetry pipeline does:**
   The ESP32 reads 4 analog pins (Probes 1 & 2, Water Level, Rain Pin) and DHT22 on GPIO 4 every 2.5 seconds, emitting plain JSON lines over USB serial (115200 baud). The Python host background thread reads lines, updates in-memory latest state, and logs valid records into SQLite (`data/sensor_history.db`). Timestamps represent **host receipt time** (UTC), not microcontroller measurement time.
2. **Whether a genuine four-hour, 24-step sequence can currently be constructed:**
   **No.** Current stored telemetry records are high-frequency bursts (median $\Delta t \approx 2.52$s) with multi-hour gaps. No continuous 4-hour window at 10-minute cadence exists in the database.
3. **Which of the 14 model features are available, missing, or incompatible:**
   - **Available:** 4 / 14 (`weather_temp`, `weather_humidity`, `hour`, `month`).
   - **Incompatible Scale:** 2 / 14 (`weather_rain` binary vs mm, `soil_moisture` relative % vs volumetric %).
   - **Missing on Hardware:** 8 / 14 (`weather_pressure`, `weather_wind_speed`, `weather_radiation`, `soil_temperature_0-7cm`, `soil_temperature_7-18cm`, `ec`, `ph`, `irrigation_duration_minutes`).
4. **What calibration and hardware/data-source work is required:**
   - **Hardware additions:** Barometric pressure (BMP280), anemometer, pyranometer (or Open-Meteo weather bridge), dual-depth soil temperature probes (DS18B20 at 0–7cm and 7–18cm), and Modbus soil EC/pH probe.
   - **Calibration:** Gravimetric oven-drying core calibration to establish a soil-specific transfer function from raw ADC to true volumetric water content ($VWC$).
   - **Firmware / Daemon:** Server-side 10-minute cron daemon or ESP32 internal ring buffer to produce genuine 10-minute observations.
5. **Whether live inference is currently permitted by the input-validity checks:**
   **Strictly NO.** Live inference is rejected by `validate_and_construct_sequence()` with `HIGH_FREQUENCY_PACKET_STREAM` and `MISSING_FEATURE_COLUMNS`. The system displays:
   `LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL`.
6. **Which files were changed / created:**
   - Created: `training/experimental/phase23/sequence_buffer.py`
   - Created: `training/experimental/phase23/README.md`
   - Created: `test_phase23_sequence_buffer.py`
   - Created: `docs/phase23_live_sensor_readiness.md`
   - Unmodified: `dashboard.py`, `esp32/sensor_test/sensor_test.ino`, `database.py`, `training/inference.py`, checkpoints, database files.
7. **Which tests actually ran and their results:**
   62 tests across 5 test suites ran and passed in 8.211 seconds (`test_ml_integration.py`: 25, `test_phase14_integration.py`: 10, `test_phase15_transformer.py`: 10, `test_phase22_forecasting_integration.py`: 7, `test_phase23_sequence_buffer.py`: 10).
8. **Confirmation of Model Preservation & Safety Locks:**
   - **Zero model training or retraining occurred.**
   - Production classifier (`training/models/random_forest_11feat.pkl`, threshold `0.3270985223048995`, 11 features) remains active and unmodified.
   - `PUMP_ALLOWED = False` remains strictly enforced across all modules. Zero actuator or valve code exists.

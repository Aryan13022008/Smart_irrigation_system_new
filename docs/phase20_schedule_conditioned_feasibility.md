# Phase 20 — Schedule-Conditioned Forecasting Feasibility Audit Report

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Timestamp:** `2026-10-10T21:35:00`  
**Phase:** Phase 20 — Schedule-Conditioned Forecasting Feasibility Audit  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model, Threshold: `0.3271`)  
**Hardware Status:** ESP32 Dual Capacitive Probes (GPIO 32, GPIO 33), DHT22 — `PUMP_ALLOWED = False` Strictly Enforced  

---

## 1. Executive Summary & Audit Mandate

This phase performs a **read-only feasibility and research-design audit** investigating whether adding planned irrigation schedule information could improve the experimental 24-hour soil-moisture forecasting system.

### Core Audit Discoveries
1. **Planned Schedule Information Does Not Exist in the Dataset:**  
   A comprehensive inventory across all raw, merged, and preprocessed tables confirms that the repository contains **zero planned irrigation start times, zero planned durations, zero recurring timer configurations, and zero farm automation calendar entries**. All existing actuator timestamps in `raw_actuators.csv` and `final_irrigation_events.csv` represent **realized historical telemetry**, recorded after the physical valves moved.
2. **`water_vol_to_24h` is a Realized Outcome, Not an Advance Schedule:**  
   The preprocessed feature `water_vol_to_24h` represents the volume of water applied over rows $t$ through $t+143$. As explicitly documented by the dataset authors in `README.md` and `DATA_DICTIONARY.md`, using this column as an input in a live predictive setting is **severe forward-looking leakage**.
3. **Phase 19 Causal Selector Verified as Causal and Unbiased:**  
   The 4-hour rule implemented in `training/experimental/phase19/regime_prototype.py` (`past_water_4h > 0`) uses strictly backward-looking valve records ($t-23$ to $t$). No future water volume, target moisture, or retrospective event label enters the live selector.
4. **Current Test Set Cannot Validate Schedule Conditioning:**  
   In the standard chronological split ($70\% / 15\% / 15\%$), the test set (late August to late September 2025) contains **exactly 0 irrigation events**. Evaluating a schedule-conditioned model on unseen test events is mathematically impossible with the current dataset partitions without re-partitioning or acquiring multi-season field data.
5. **Decision Verdict:**  
   **GO for schedule-data collection and experiment protocol design, but STRICT NO-GO for schedule-conditioned model training or production deployment.**

---

## 2. Task 1: Inventory of Available Schedule Information

An exhaustive file-by-file audit of all telemetry, events, registries, and documentation in `/Users/aryanagrawal/ESA_PROJECT`:

| Data Category | Existence in Project | Source File Path | Nature / Classification | Availability at Forecast Origin $t$ | Reliability & Limitations |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Planned Start Times** | **NO** | None | Absent | **Unavailable** | No advance schedule entries exist. |
| **Planned Durations / Vol** | **NO** | None | Absent | **Unavailable** | No planned duration tables exist. |
| **Recurring Timers / Config** | **NO** | None | Absent | **Unavailable** | No timer or cron configuration files exist. |
| **Farm Automation Calendar** | **NO** | None | Absent | **Unavailable** | No agronomic management log exists. |
| **Actual Actuator Telemetry** | **YES** | `01_raw_data/raw_irrigation_actuators/raw_actuators.csv` | Observed/Synthetic minute-level binary transitions | Past records ($\le t$) available; Future ($> t$) **unavailable** | 3,055,944 rows; 19.1% `is_raw=True`, 80.9% platform synthetic interpolation. Naive local Europe/Rome time. |
| **Reconstructed Event Log** | **YES** | `02_processed_data/irrigation_events/final_irrigation_events.csv` | Post-hoc heuristically reconstructed events | **Unavailable at $t$** (computed post-season) | 789 events; reconstructed using 4-phase heuristic filtering. Coupling: synthetic events accepted only if soil moisture was low. |
| **Forward Water Volume** | **YES** | `02_processed_data/preprocessed/dataset_zone_{1..5}_preprocessed.csv` | Retrospective realized forward sum over $[t, t+143]$ | **Unavailable at $t$** (outcome of future valve openings) | Column `water_vol_to_24h`. Legitimate only for retrospective evaluation stratification; strict leakage if used as a live input. |
| **Backward Water Volume** | **YES** | `02_processed_data/preprocessed/dataset_zone_{1..5}_preprocessed.csv` | Retrospective backward sum over $[t-24, t]$ | **Available at $t$** | Column `water_vol_past_4h`. Causal and valid. |
| **Historical Weather Data** | **YES** | `02_processed_data/preprocessed/dataset_zone_{1..5}_preprocessed.csv` | Open-Meteo ERA5 Reanalysis | Past records ($\le t$) available; Future ($> t$) **unavailable** | Global atmospheric satellite/station assimilation, NOT numerical weather prediction (NWP) forecasts issued at time $t$. |

---

## 3. Task 2: Forecast Information Boundary

Defining the prediction origin as time $t$, with a forecasting target of mean soil moisture over $[t, t+24\text{ hours}]$:

### 3.1 Information-Availability Matrix

| Candidate Input | Available at Time $t$? | Allowed for Live Prediction? | Temporal Classification | Governance Conditions & Rules |
| :--- | :---: | :---: | :--- | :--- |
| **Current soil moisture ($\text{soil\_moisture}[t]$)** | **YES** | **YES** | Observed at $t$ | Must be physically sampled by or at time $t$. |
| **Historical soil moisture ($t-4\text{h}$ to $t$)** | **YES** | **YES** | Observed at $t$ | Strictly causal; no records after $t$. |
| **Past irrigation duration ($\text{past\_water\_4h}$)** | **YES** | **YES** | Observed at $t$ | Strictly causal historical valve open time $\le t$. |
| **Ambient weather history ($\le t$)** | **YES** | **YES** | Observed at $t$ | Sensor or weather observations prior to $t$. |
| **Planned irrigation start time** | **NO (Missing)** | **YES (Hypothetical)** | Plan known at $t$ | Allowed ONLY if an immutable schedule version was published before $t$. |
| **Planned irrigation duration / volume** | **NO (Missing)** | **YES (Hypothetical)** | Plan known at $t$ | Allowed ONLY if pre-published; must NEVER substitute actual realized duration. |
| **Actual future irrigation duration ($> t$)** | **NO** | **NO (STRICT LEAKAGE)** | Outcome after $t$ | Realized physical execution. Unconditionally prohibited as a live input. |
| **Forward water volume (`water_vol_to_24h`)** | **NO** | **NO (STRICT LEAKAGE)** | Outcome after $t$ | Retrospective realization. Valid exclusively for post-hoc evaluation slicing. |
| **Future soil-moisture observations ($> t$)** | **NO** | **NO (TARGET)** | Outcome after $t$ | Ground-truth targets or retrospective evaluation metrics only. |
| **Retrospective event labels (e.g. `#639–#642`)** | **NO** | **NO (LEAKAGE)** | Outcome after $t$ | Generated post-season; prohibited as a live routing input. |
| **Realized future weather (ERA5 $> t$)** | **NO** | **NO (LEAKAGE)** | Outcome after $t$ | Reanalysis of future weather. In live operation, only NWP forecasts issued at $t$ may be used. |

### 3.2 Explicit Temporal Separation
1. **Observed at Time $t$:** Telemetry and sensor states up to index $i$ ($t$).
2. **Plans Known at Time $t$:** Pre-published control schedules, timer tables, automation calendar entries (**completely absent from the dataset**).
3. **Outcomes Known Only After Time $t$:** Realized valve openings, delivered volumetric pulses, percolation curves, weather realizations, target $y_{t+24\text{h}}$.

---

## 4. Task 3: Audit of the Existing Phase 19 Selector

The causal prototype implemented in [`training/experimental/phase19/regime_prototype.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase19/regime_prototype.py) was audited against all governance rules:

### 4.1 Implementation Audit
- **Causal Rule:**
  ```python
  def causal_regime_gate(input_sequence_df_or_arr):
      past_valve = float(input_sequence_df_or_arr["irrigation_duration_minutes"].sum())
      return "TRANSFORMER" if past_valve > 0 else "PERSISTENCE"
  ```
- **Boundary Verification:** The input sequence spans exclusively rows $i-23$ to $i$ ($t-230\text{ min}$ to $t$). No future rows ($> i$) are touched.
- **Independence from Leakage:** No future water volume (`water_vol_to_24h`), target moisture, or retrospective event tags enter `causal_regime_gate()`.
- **Forecast Alignment:** Identical forecast origins $i$ and identical targets $y_i$ were evaluated across Persistence, Transformer, Causal Gate, and Oracle.
- **Metrics Reproduction:** Pooled sample-level errors on the validation split ($N = 6,326$) were verified:
  - Persistence alone: $\text{MAE} = 1.2502\%$
  - Transformer alone: $\text{MAE} = 4.0465\%$
  - Causal Online Prototype: $\text{MAE} = \mathbf{1.2048\%}$
  - Retrospective Oracle: $\text{MAE} = \mathbf{1.0073\%}$
- **Reporting Clarity:** In Phase 19, event-stratified metrics are explicitly designated as **retrospective analysis slices**, not live selector performance.

---

## 5. Task 4: Design of Future Schedule-Conditioned Experiment (No Training)

Four candidate approaches are formulated for future research:

### Approach A: Naive Persistence Baseline
- **Inputs:** Current soil moisture $\text{soil\_moisture}[t]$.
- **Retraining:** Zero parameters; no training required.
- **Role:** The benchmark to beat during quiescent static periods ($\text{MAE} = 0.412\%$ on the test set).
- **Failure Mode:** Massive error when wetting fronts occur ($\text{MAE} > 3.6\%$).

### Approach B: Existing Un-Gated Transformer
- **Inputs:** 14 historical environmental features ($t-4\text{h}$ to $t$).
- **Retraining:** Reuses saved weights `transformer_experimental.pt`.
- **Role:** High performance during active wetting dynamics ($\text{MAE} = 0.712\%$).
- **Failure Mode:** Over-predicts non-existent oscillations during quiescent dry periods ($\text{MAE} = 6.124\%$).

### Approach C: Future Schedule-Conditioned Model
- **Inputs:** 14 historical environmental features + Schedule Vector:
  - `planned_irrigation_scheduled` ($\{0, 1\}$)
  - `planned_lead_time_hours` (hours until scheduled opening, e.g. $4.5\text{ h}$)
  - `planned_duration_minutes` (planned duration, e.g. $15\text{ min}$)
  - `planned_volume_liters` (planned nominal delivery)
- **Retraining Requirement:** **MANDATORY.** Existing checkpoints accept only `in_dim = 14`. A schedule-conditioned architecture requires a new feature embedding or cross-attention conditioning layer.
- **Information Acquisition:** Must be received from farm automation calendar or timer registry prior to forecast origin $t$.
- **Expected Failure Modes:**
  - Cancelled or postponed schedules (model anticipates wetting front that never arrives).
  - Actuation latency or pressure drops delivering less volume than scheduled.
  - Probe placement outside the emitter wetting bulb.
- **Fair Baseline Comparison:** Must be benchmarked against a **Schedule-Aware Hydraulic Lookup Baseline** (e.g., Persistence when no schedule exists, and an empirical hydraulic step-function when a schedule exists).

### Approach D: Causal Hybrid Prototype
- **Inputs:** Historical sequence ($t-4\text{h}$ to $t$) and causal gating flag (`past_water_4h > 0` or valid pre-published schedule flag).
- **Retraining Requirement:** None for rule-based gating; reuses existing checkpoints.
- **Role:** Combines persistence during quiescent periods with deep learning during active percolation.

---

## 6. Task 5: Defensible Evaluation Protocol

A scientifically defensible evaluation protocol for future phases must satisfy:

1. **Multi-Horizon Evaluation:**
   Forecast horizons evaluated at **$6\text{ hours}$**, **$12\text{ hours}$**, and **$24\text{ hours}$** to capture initial infiltration, peak wetting front, and subsequent redistribution.
2. **Chronological Event-Rich Partitioning:**
   - Training, validation, and test splits must be strictly chronological.
   - **Crucial Dataset Limitation:** The existing chronological test set (late August to late September 2025) has **ZERO irrigation events**.
   - Evaluating a schedule-conditioned model on the current test partition is impossible because there are no test events to forecast.
   - **Prohibition:** Moving training-period events into the test set while claiming they are "unseen" is **strictly rejected as scientific data snooping**.
3. **Regime-Stratified Reporting:**
   Every experiment must report metrics across three distinct slices:
   - Overall Performance ($N_{\text{total}}$)
   - Quiescent Dry Performance ($N_{\text{quiescent}}$)
   - Active Irrigation Performance ($N_{\text{active}}$)
4. **Cautious Metric Interpretation:**
   Within individual sectors, target variance during plateaus is small ($\sigma < 0.8\%$). In such slices, $R^2$ will naturally be negative even when absolute error ($\text{MAE}$) is small. Reports must focus primarily on **$\text{MAE}$ and $\text{RMSE}$**.
5. **Cluster-Block Bootstrap Uncertainty:**
   Overlapping 10-minute rolling windows share $95.8\%$ of input features and $99.3\%$ of target steps. All uncertainty intervals must be computed via **Cluster-Block Bootstrap** resampled at the independent event or calendar-day level ($1,000$ resamples), not assuming independent sample draws.

---

## 7. Task 6: Data Requirements & Collection Plan

Because planned schedules do not exist in the current dataset, future field trials must collect a dedicated, immutable schedule registry:

### 7.1 Pre-Event Fields (Known at Forecast Origin $t$)
- `schedule_id`: Unique identifier (UUID).
- `sector_id`: Target irrigation zone ($1–5$).
- `schedule_created_at`: Exact timestamp when the schedule was published/committed.
- `planned_start_at`: Scheduled valve opening timestamp.
- `planned_end_at`: Scheduled valve closing timestamp.
- `planned_duration_minutes`: Scheduled valve opening duration.
- `planned_volume_liters`: Planned water quota.
- `schedule_status`: Current lifecycle state (`PLANNED`, `ACTIVE`, `MODIFIED`, `CANCELLED`).
- `schedule_version`: Incremental version number to track schedule revisions.
- `timestamp_timezone`: Explicit timezone (`Europe/Rome` / ISO 8601 with offset).

### 7.2 Post-Event Execution Fields (Known Only After Time $t$)
- `actual_start_at`: Hardware-observed valve opening timestamp.
- `actual_end_at`: Hardware-observed valve closing timestamp.
- `actual_duration_minutes`: Telemetry-measured opening duration.
- `actual_volume_liters`: In-line flow meter volume reading.
- `execution_discrepancy_reason`: `NORMAL`, `MANUAL_OVERRIDE`, `PRESSURE_DROP`, `RAIN_INTERRUPT`.
- `soil_moisture_before`: Probe reading at `actual_start_at`.
- `soil_moisture_after`: Probe reading at peak infiltration ($t + 2\text{h}$).
- `sensor_timestamp`: Sensor acquisition timestamp.

---

## 8. Final Decision & Verification Summary

### Decision Verdict
- **GO:** For schedule data collection schema implementation and future experiment design.
- **STRICT NO-GO:** For schedule-conditioned model training, checkpoint modification, or production deployment.

### Evidence Still Missing
1. **No Planned Schedule Data:** The current repository contains zero advance schedule records.
2. **No Independent Event-Rich Test Data:** The existing chronological test split is 100% quiescent post-season.
3. **No Volumetric Flow-Meter Ground Truth:** Actuators only report binary open/close timers; actual delivered volume is unmetered.
4. **Physical Sensor Mismatch:** Live ESP32 capacitive probes remain uncalibrated to soil volumetric water content.

### Production Posture
- Production model: `training/models/random_forest_11feat.pkl` (Phase 11, threshold `0.3271`) remains sole active advisory model.
- Pump safety hard-lock: `PUMP_ALLOWED = False` remains strictly enforced.

---

## 9. Verification & Regression Status

The full unit and integration test suite was executed:
- **Command:** `python3 -m unittest discover -s . -p "test_*.py"`
- **Result:** **45 / 45 tests PASSED (100%)** in 8.803s.
  - `test_ml_integration.py`: 25 passed.
  - `test_phase14_integration.py`: 10 passed.
  - `test_phase15_transformer.py`: 10 passed.

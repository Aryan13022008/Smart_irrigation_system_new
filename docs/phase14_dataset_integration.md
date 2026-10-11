# Smart Agriculture — Phase 14: Dataset Integration & Experimental Model Audit Report

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 14 — Integrate and Improve Existing Smart Irrigation Project Using New Dataset  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model, Threshold: 0.3271)  
**Hardware Status:** ESP32 (GPIO 32, GPIO 33 capacitive probes, DHT22) — Automatic Pump Actuation STRICTLY DISABLED  

---

## 1. Summary of Files Added and Modified

| File Path | Action | Description / Rationale |
|:---|:---:|:---|
| [`training/experimental/phase13/lstm_experimental.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/lstm_experimental.pt) | Added | Experimental 1-layer LSTM forecasting checkpoint (6,689 params) with embedded scaler parameters and feature schemas. |
| [`training/experimental/phase13/tcn_experimental.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/tcn_experimental.pt) | Added | Experimental 3-block Dilated Causal TCN forecasting checkpoint (17,921 params). |
| [`training/experimental/phase13/phase13_lstm_tcn_results.json`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/phase13_lstm_tcn_results.json) | Added | Training provenance, history, and test metrics for Phase 13 experimental models. |
| [`training/experimental/phase13/phase13_lstm_tcn_results.md`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/phase13_lstm_tcn_results.md) | Added | Human-readable Phase 13 experimental training summary. |
| [`training/experimental/phase13/forecasting_models.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/forecasting_models.py) | Created | Standalone PyTorch module providing architecture definitions, model loading, preprocessing, inference, and multi-zone dataset evaluation. |
| [`test_phase14_integration.py`](file:///Users/aryanagrawal/ESA_PROJECT/test_phase14_integration.py) | Created | 10 unit and integration tests covering model loading, feature order, sequence dimensions, chronological split integrity, persistence baseline reproduction, zone isolation, pump lock, and production isolation. |
| [`dashboard.py`](file:///Users/aryanagrawal/ESA_PROJECT/dashboard.py) | Modified | Added a dedicated, non-intrusive **Experimental 24-Hour Soil-Moisture Forecasting** research section to the AI Models page. Preserved all production cards, navigation, and tests. |
| [`docs/phase14_dataset_integration.md`](file:///Users/aryanagrawal/ESA_PROJECT/docs/phase14_dataset_integration.md) | Created | This authoritative Phase 14 audit, integration, and verification report. |

---

## 2. Dataset Provenance & Target Audit Findings

### 2.1 Origin and Physical Setup
- **Dataset Title:** *Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial (Arnesano, Apulia, Italy, 2025)*.
- **Location:** Arnesano, Province of Lecce, Apulia, Italy ($40.3493^\circ\text{ N}, 18.0815^\circ\text{ E}$).
- **Period:** February 9, 2025 to September 27, 2025 ($33,259$ raw 10-minute grid steps).
- **Sectors / Crops:**
  - Sector 1: Tomato, open field (9,531 preprocessed rows)
  - Sector 2: Tomato, open field (11,929 preprocessed rows)
  - Sector 3: Tomato, pots (1,118 rows — probe malfunctioned in June 2025; no valid contiguous test windows)
  - Sector 4: Zucchini (12,676 preprocessed rows)
  - Sector 5: Blueberry (11,814 preprocessed rows)
- **Actuators & Sensors:** Soil moisture probes (`SUT*`, 0–100% relative scale), soil EC (`SNPK*_CE`), soil pH (`SNPK*_PH`), outside/inside temperature (`STE*`, `STI*`), outside/inside humidity (`SUE*`, `SUI*`), and irrigation valves (`AIRR*`). Weather data is ERA5 reanalysis from the Open-Meteo API.

### 2.2 Target Formulation: `target_mean_24h` and `target_point_24h`
In [`03_code/02_preprocessing.py#L129-L137`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/03_code/02_preprocessing.py#L129-L137):
- Horizon: $24\text{ hours} = 144$ steps of 10 minutes.
- `target_point_24h`: `df["soil_moisture"].shift(-144)` (instantaneous soil moisture exactly 24 hours ahead).
- `target_mean_24h`: `df["soil_moisture"].rolling(window=forward, min_periods=min_samples).mean()` where `forward = FixedForwardWindowIndexer(window_size=144)`.
- **Target Alignment:** The rolling window computes the forward mean over $[t, t+143]$.

### 2.3 Status of Valve/Actuator Labels
- The raw telemetry records physically observed valve signals (`is_raw = True`) and platform-inferred synthetic states (`is_raw = False`).
- The pipeline applies a heuristic 4-phase reconstruction: pairing open/close transitions, requiring soil moisture below start thresholds for synthetic openings, debouncing (< 2 min), discarding short pulses (< 15 min), and filtering long anomalies (> 5 hours).
- **Finding:** Actuator signals are **partially observed, partially reconstructed with heuristics**. They cannot be treated as unquestioned ground truth.

---

## 3. Integration of Experimental LSTM and TCN Artifacts

The experimental artifacts have been integrated into [`training/experimental/phase13/`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/):
1. **Model Checkpoints:** Both `lstm_experimental.pt` and `tcn_experimental.pt` load cleanly in PyTorch `eval()` mode.
2. **Feature Ordering:** Exactly enforces the 14 ordered features:
   `weather_temp`, `weather_humidity`, `weather_rain`, `weather_pressure`, `weather_wind_speed`, `weather_radiation`, `soil_temperature_0-7cm`, `soil_temperature_7-18cm`, `ec`, `ph`, `soil_moisture`, `irrigation_duration_minutes`, `hour`, `month`.
3. **Preprocessing & Scaler:** `forecasting_models.py` uses the exact 14-element `scaler_mean` and `scaler_scale` vectors fitted on the training split rows.
4. **Safety Isolation:** `PUMP_ALLOWED = False` is enforced at the module level. Under no circumstances can forecasting predictions trigger pump relays.

---

## 4. Baseline vs Model Results on Identical Test Windows

All models were evaluated on the **exact $6,348$ contiguous 10-minute held-out test windows** across Sectors 1, 2, 4, and 5:

### 4.1 Overall Metrics Benchmark

| Model | Test MAE (%) | Test RMSE (%) | Pooled Test $R^2$ | Number of Parameters |
|:---|:---:|:---:|:---:|:---:|
| **Naive Persistence Baseline** ($\hat{y} = \text{soil\_moisture}[t]$) | **0.4124** | **0.7641** | **0.99863** | **0 (Heuristic)** |
| **Experimental LSTM** | **3.8665** | **5.1434** | **0.93794** | 6,689 |
| **Experimental TCN** | **4.6904** | **7.6977** | **0.86098** | 17,921 |

### 4.2 Inconsistency Resolution: Why Baseline Wins & Why $R^2$ is Misleading

1. **Stationarity of the Test Set (Zero Future Irrigation):**
   - In the test split (the final 15% of each sector, spanning late August to late September 2025), **future irrigation was 0.0% across all sectors** ($0$ out of $6,348$ windows had irrigation).
   - Because no irrigation occurred, soil moisture was in natural static equilibrium.
   - The current moisture was nearly identical to the 24-hour future mean, allowing the persistence baseline to achieve a tiny MAE of **$0.4124\%$**.
   - The neural networks, having learned dynamic irrigation response weights during training (February–August), over-predicted fluctuations in the stationary post-season, resulting in $9\times$ to $11\times$ higher errors ($3.867\%$ and $4.690\%$).

2. **Simpson's Paradox / Ecological Fallacy in Pooled $R^2$:**
   - In the pooled dataset, the variance of $y$ is massive ($Var(y) \approx 780$) because Sectors 1 & 2 operate near $15–20\%$, Sector 4 operates near $57\%$, and Sector 5 operates near $2\%$.
   - Any model that roughly separates the sectors receives an artificially high pooled $R^2$ ($> 0.86$).
   - However, within any individual sector, the target standard deviation in the test split is tiny ($0.16\%$ in Sector 1, $0.26\%$ in Sector 2). Within each sector, the neural networks' $R^2$ is **massively negative** ($-7$ to $-2864$).

---

## 5. Per-Zone Evaluation Breakdown

| Sector | Crop & System | Test Windows | Sector Target Mean | Sector Target Std | Persistence MAE (%) | LSTM MAE (%) | TCN MAE (%) | Within-Sector $R^2$ (LSTM / TCN) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sector 1** | Tomato, Open Field | 1,408 | 19.27% | 0.16% | **0.4678** | 1.4221 | 2.2372 | -525.51 / -2864.31 |
| **Sector 2** | Tomato, Open Field | 1,721 | 15.54% | 0.26% | **0.3810** | 2.6545 | 2.1904 | -158.69 / -94.41 |
| **Sector 3** | Tomato, Pots | 0 | — | — | N/A (Failed probe) | N/A | N/A | N/A |
| **Sector 4** | Zucchini | 1,599 | 57.40% | 1.95% | **0.3807** | 6.2999 | 1.6669 | -13.08 / -0.18 |
| **Sector 5** | Blueberry | 1,620 | 2.19% | 2.06% | **0.4290** | 4.8766 | 12.4626 | -7.13 / -40.26 |

**Conclusion:** The persistence baseline outperforms both LSTM and TCN in **every single sector without exception**.

---

## 6. Model Improvement Evaluation

- **Did Any Complex Model Improve Performance?** **NO.**
- Tabular Gradient Boosting (`HistGradientBoostingRegressor`) was also evaluated on the dataset:
  - Tabular HGB Regressor: Test MAE **$7.708\%$**
  - Delta Ridge Model ($\hat{\Delta} = y - \text{moisture}$): Test MAE **$1.666\%$**
  - Persistence Baseline ($\hat{\Delta} = 0$): Test MAE **$0.412\%$**
- **Decision:** In accordance with empirical machine learning principles, **the simplest model (persistence baseline) is retained as the authoritative reference**. The neural networks must not be deployed to production.

---

## 7. Automated Test Suite Execution

Both test suites were executed directly on the project:

```bash
python3 -m unittest test_ml_integration.py
python3 -m unittest test_phase14_integration.py
```

### Execution Results:
1. `test_ml_integration.py`: **25 / 25 PASS (100%)**
   - Confirmed Phase 11 11-feature model (`random_forest_11feat.pkl`, threshold $0.3271$) remains active.
   - Confirmed zero model-switching selectboxes exist.
   - Confirmed dashboard AppTest passes cleanly across all pages.
2. `test_phase14_integration.py`: **10 / 10 PASS (100%)**
   - Verified artifact integrity, checkpoint loading, parameter counts, feature ordering, and positive scalers.
   - Verified chronological split integrity and zero future leakage.
   - Verified persistence baseline reproduction ($0.4124$ MAE) on identical $6,348$ windows.
   - Verified Simpson's paradox and negative within-zone $R^2$.
   - Verified `PUMP_ALLOWED = False` safety lock.

**Total Verified Tests:** **35 / 35 PASS (100%)**.

---

## 8. How to Run the Experiment

### A. Run Standalone Dataset Benchmark:
```bash
python3 training/experimental/phase13/forecasting_models.py
```

### B. Run Phase 14 Unit Test Suite:
```bash
python3 -m unittest test_phase14_integration.py
```

### C. Launch the Dashboard to View the Research Card:
```bash
streamlit run dashboard.py
```
Navigate to **🤖 AI Models** $\to$ scroll down to **🧪 Experimental 24-Hour Soil-Moisture Forecasting (Research)**.

---

## 9. Limitations & Recommended Next Steps

1. **Task Boundary:** 24-hour soil-moisture forecasting ($\%$) is fundamentally distinct from an irrigation decision (ON/OFF). Do not convert soil-moisture forecasts into automatic actuator triggers without agronomically validated matric potential thresholds.
2. **Hardware Incompatibility:** The experimental models require 14 inputs, including soil EC, pH, solar radiation, and dual-depth soil temperatures. The ESP32 station lacks these probes. Do not fabricate missing sensors for live inference.
3. **Safety Status:** Automatic pump actuation remains hard-locked to disabled (`PUMP_ALLOWED = False`). All predictions are advisory only.

# Smart Agriculture — Phase 17: Irrigation Event Validation & Forecasting Audit

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 17 — Irrigation Event Validation and Forecasting Audit  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model, Threshold: `0.3271`)  
**Hardware Status:** ESP32 Dual Capacitive Probes (GPIO 32, GPIO 33), DHT22 — `PUMP_ALLOWED = False` Strictly Enforced  

---

## 1. Executive Summary & Core Discoveries

This audit independently investigates the scientific validity, target formulation, event distributions, and split methodology of the 24-hour soil-moisture forecasting models developed across Phases 13 through 16.

### Key Audit Findings

1. **Reconciliation of Test Window Counts (6,347 vs 6,348):**
   The 1-window difference between Phase 13 ($N = 6,347$) and Phases 14–16 ($N = 6,348$) is fully resolved. It stems from floating-point integer truncation when partitioning Sector 4 (`int(n * 0.70) + int(n * 0.15)` vs `int(n * 0.85)`). Both evaluate identical distributions with MAE differences $< 0.0006\%$.
2. **Target Formulation & Inclusion of Current Observation:**
   The target `target_mean_24h` is computed via `FixedForwardWindowIndexer(window_size=144)`. In pandas, this rolling window covers $[t, t+143]$. Consequently, the target **includes the current observation** $\text{soil\_moisture}[t]$ as $1$ of the $144$ averaged points (a weight of $1/144 \approx 0.694\%$). Feature sets (`EXPERIMENTAL_FEATURES`) are strictly causal (steps $t-23$ to $t$) and properly exclude leaking future fields (`water_vol_to_24h`, `real_moisture_delta`).
3. **The Zero-Irrigation Test Split Mechanism:**
   In the standard chronological test split ($N = 6,348$ across Sectors 1, 2, 4, and 5), there are **EXACTLY 0 (ZERO) irrigation events**. The naive 70%/15%/15% chronological split placed the test set entirely into the late-season post-harvest tail (late August to late September 2025), after irrigation valves were permanently closed.
4. **Resolution of the Persistence Baseline "Super-Human" Performance:**
   The apparent superiority of the persistence baseline ($\text{MAE} = 0.412\%$) on the test set is an artifact of evaluating exclusively on a quiescent, un-irrigated regime where soil moisture changes by fractions of a percent over 24 hours.
   - **During Quiescent Dry Periods (Test Set, N=6,348):** Persistence achieves $\text{MAE} = 0.412\%$, whereas neural networks over-predict fluctuations ($\text{MAE} = 3.867\% - 6.124\%$).
   - **During Active Future Irrigation (Validation Set, N=527):** Persistence error **spikes to $3.629\%$**, while the **Transformer achieves $\text{MAE} = 0.712\%$** (beating persistence by $5\times$), the **LSTM achieves $\text{MAE} = 2.473\%$**, and the **TCN achieves $\text{MAE} = 2.616\%$**.
   When irrigation actually occurs, the deep learning sequence models decisively outperform persistence.
5. **Actuator Label Reconstruction & Heuristics:**
   Raw valve telemetry contains only $19.1\%$ physically observed states (`is_raw = True`); $80.9\%$ are platform-inferred synthetic states. Crucially, the preprocessing pipeline accepted synthetic openings only when $\text{soil\_moisture} < \text{threshold\_start}$, introducing circular coupling between the sensor and actuator labels.
6. **Production Hardware Mismatch:**
   The dataset features commercial probes calibrated to field soil matric potentials (wilting point 20%, field capacity 60–80%). The live ESP32 capacitive probes use a 2-point air/water index (dry air 0%, submerged water 100%) that is physically decoupled from soil volumetric water content. Production pump actuation must remain permanently locked (`PUMP_ALLOWED = False`).

---

## 2. Task 1: Reconciliation of Evaluation Metrics Across Phases

### 2.1 Cross-Phase Metrics Comparison

| Phase | Evaluation Scope | Test Windows ($N$) | Persistence MAE (%) | LSTM MAE (%) | TCN MAE (%) | Transformer MAE (%) | Pooled $R^2$ (LSTM / TCN / TF) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 13** | Initial LSTM & TCN training | 6,347 | 0.4124 | 3.8670 | 4.6909 | — | 0.9379 / 0.8610 / — |
| **Phase 14** | Multi-zone integration | 6,348 | 0.4124 | 3.8665 | 4.6904 | — | 0.9379 / 0.8610 / — |
| **Phase 15** | Transformer forecasting | 6,348 | 0.4124 | 3.8665 | 4.6904 | 6.1237 | 0.9379 / 0.8610 / 0.8222 |
| **Phase 16** | Local Mac audit & execution | 6,348 | 0.4124 | 3.8665 | 4.6904 | 6.1237 | 0.9379 / 0.8610 / 0.8222 |

### 2.2 Root Cause of Window Count Difference ($6,347$ vs $6,348$)
In [`03_code/config.py`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/03_code/config.py) and Phase 13 training scripts, splits were calculated as:
```python
# Phase 13 formulation:
train_df = df.iloc[: int(n * 0.70)]
val_df = df.iloc[int(n * 0.70) : int(n * 0.85)]
test_df = df.iloc[int(n * 0.85) :]
```
In Phase 14 ([`training/experimental/phase13/forecasting_models.py#L323-L325`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/forecasting_models.py#L323-L325)):
```python
# Phase 14-16 formulation:
n_train = int(n * 0.70)
n_val = int(n * 0.15)
test_df = df.iloc[n_train + n_val :]
```
For Sector 4 ($N = 12,676$ rows):
- $12,676 \times 0.70 = 8,873.2 \to 8,873$
- $12,676 \times 0.15 = 1,901.4 \to 1,901$
- $n_{\text{train}} + n_{\text{val}} = 8,873 + 1,901 = 10,774$
- However, $12,676 \times 0.85 = 10,774.6 \to 10,774$. 
In Sector 1 ($N = 9,531$ rows):
- $9,531 \times 0.70 = 6,671.7 \to 6,671$
- $9,531 \times 0.15 = 1,429.65 \to 1,429$
- $n_{\text{train}} + n_{\text{val}} = 8,100$
- But $9,531 \times 0.85 = 8,101.35 \to 8,101$
Because $8,100 \ne 8,101$, `test_df` in Phase 14–16 started at index 8,100 instead of 8,101, admitting exactly **one additional valid 24-step window** in Sector 1 (1,408 vs 1,407). Both test sets represent the identical physical post-season partition, with metric discrepancies strictly $< 0.0006\%$.

---

## 3. Task 2: Target & Persistence Baseline Audit

### 3.1 Construction of `target_mean_24h`
In [`03_code/02_preprocessing.py#L130-L136`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/03_code/02_preprocessing.py#L130-L136):
```python
forward = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)
df[f"target_mean_{horizon}h"] = df["soil_moisture"].rolling(
    window=forward, min_periods=min_samples
).mean()
```
Where `steps = 144` (144 intervals of 10 minutes = 24 hours).

#### Mathematical Window Formulation
In pandas, `FixedForwardWindowIndexer(window_size=144)` computes a forward window starting at the current row $i$:
$$y_i = \frac{1}{M} \sum_{k=0}^{143} \text{soil\_moisture}[i + k]$$
(where $M \ge 72$ is the count of non-null observations in the window).

#### Audit Questions Resolved
1. **Does the target include the current observation?**
   **YES.** The summation starts at $k = 0$, which is index $i$ ($\text{soil\_moisture}[i]$). Because the window contains 144 steps, the current observation contributes exactly $\frac{1}{144} \approx 0.694\%$ to the target mean.
2. **Does persistence use the same target rows?**
   **YES.** Both the persistence baseline ($\hat{y}_i = \text{soil\_moisture}[i]$) and the neural models ($\hat{y}_i = f(X_{i-23:i})$) are evaluated against the identical target value $y_i$.
3. **Is there future-data leakage in feature sequences?**
   **NO.** Input sequences $X$ span rows $i-23$ to $i$. The forward-looking water volume `water_vol_to_24h` and diagnostic target `real_moisture_delta` are strictly excluded from `EXPERIMENTAL_FEATURES`. All 14 input features are strictly causal.
4. **Is persistence legitimate or leaking?**
   Predicting tomorrow's average moisture using today's current moisture ($\hat{y} = x_t$) is a classic, legitimate zero-parameter time-series heuristic. It does not access future records. However, because $x_t$ is part of the 24-hour mean, in a stationary regime where $x_{t+k} \approx x_t$, the persistence heuristic is mathematically near-optimal.

---

## 4. Task 3: Characterization of Irrigation Events

Using the dataset's validated event registry ([`final_irrigation_events.csv`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/02_processed_data/irrigation_events/final_irrigation_events.csv)):

### 4.1 Per-Sector Event Inventory & Soil-Moisture Dynamics

| Sector | Crop & System | Total Events | Std. Irrigation | Fertigation | Initial Moisture (%) | Final Moisture (%) | Mean Delta (%) | Positive Delta Share | Dominant Termination Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sector 1** | Tomato, Field | 87 | 45 | 42 | $39.2 \pm 10.4$ | $46.3 \pm 14.1$ | $+7.07 \pm 12.7$ | 50.6% (44/87) | Fertigation timeout (45) |
| **Sector 2** | Tomato, Field | 195 | 131 | 64 | $38.0 \pm 7.4$ | $40.7 \pm 9.4$ | $+2.67 \pm 8.5$ | 33.8% (66/195) | Physical valve closure (89) |
| **Sector 3** | Tomato, Pots | 72 | 8 | 64 | $83.5 \pm 83.1$ | $91.3 \pm 83.8$ | $+7.74 \pm 11.3$ | 61.1% (44/72) | Fertigation timeout (63) |
| **Sector 4** | Zucchini | 289 | 197 | 92 | $77.7 \pm 11.4$ | $80.4 \pm 10.9$ | $+2.64 \pm 7.7$ | 45.3% (131/289)| Physical valve closure (146) |
| **Sector 5** | Blueberry | 146 | 51 | 95 | $68.3 \pm 16.7$ | $76.6 \pm 18.9$ | $+8.29 \pm 12.7$ | 46.6% (68/146) | Fertigation timeout (117) |
| **Total** | **All Sectors**| **789** | **432** | **357** | — | — | — | **44.7% (353/789)** | — |

### 4.2 Observed vs Inferred / Reconstructed Labels
1. **Raw Valve Telemetry:**
   In `raw_actuators.csv`, only **19.1%** of valve records are physically observed (`is_raw = True`). The remaining **80.9%** were synthetically interpolated by the farm's telemetry server.
2. **Circular Preprocessing Dependency:**
   To filter out spurious synthetic openings, [`00_irrigation_actuators_cleaning.py#L134-L158`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/03_code/00_irrigation_actuators_cleaning.py#L134-L158) accepted synthetic events only when:
   $$\text{moisture} < \text{threshold\_start}$$
   This creates an algorithmic coupling: **the label reconstruction algorithm used soil-moisture depletion to decide whether an event occurred**.
3. **The Probe Response Deficit (Only 44.7% Show Immediate Moisture Rise):**
   Across the 789 events:
   - 353 events (44.7%) showed $\Delta > 0$ (moisture increase).
   - 287 events (36.4%) showed $\Delta = 0$ (no detected change at valve close).
   - 149 events (18.9%) showed $\Delta < 0$ (apparent moisture decrease).
   *Agronomic explanation:*
   - **Fertigation pulses (15 min):** The delivered volume is small ($15 \text{ min} \times 1.1 \text{ L/min} \approx 16.5 \text{ L}$ distributed across drip lines), and often does not immediately wet the probe at 10–18 cm depth.
   - **Infiltration latency:** Soil matric suction and unsaturated hydraulic conductivity delay water percolation into the sensor zone by 1 to 4 hours after valve closure.
   - **Sensor positioning:** Probes placed slightly outside the drip emitter wetting bulb miss shallow or localized water fronts.

---

## 5. Task 4: Defensible Evaluation Feasibility Assessment

### 5.1 The Seasonality Paradox of the Dataset
A temporal audit of the preprocessed 10-minute grid reveals a severe structural partition in the data:

```
Month:       Feb      Mar      Apr      May      Jun      Jul      Aug      Sep
----------------------------------------------------------------------------------
Irrigation:  [=== Active Irrigation Period ===]           [=== Post-Season Off ==]
Sensors:     [--- Severe Data Gaps (Sparse) ---]          [-- Continuous Streams -]
```

- **February to May:** Irrigation valves operated heavily (341 events), but soil probes reported intermittently (3% coverage in February). There are **virtually zero contiguous 24-step (4-hour) windows** without missing timestamps.
- **July to August:** Sensor reporting stabilized. Thousands of contiguous windows exist with active irrigation in Sector 4 (4,670 windows) and Sector 5 (1,140 windows).
- **Late August to September:** The data stream was continuous, but **irrigation was permanently switched off**.

### 5.2 Regime-Stratified Evaluation Proof (Validation Split)
When evaluating the trained models and the persistence baseline on the **Validation Split** ($N = 6,326$), which contains both active irrigation and quiescent dry periods, the real dynamics emerge:

#### Validation Split Performance Breakdown ($N = 6,326$)

| Regime | Windows ($N$) | Persistence MAE (%) | LSTM MAE (%) | TCN MAE (%) | Transformer MAE (%) | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **During Future Irrigation** | **527** | **3.6287** | **2.4730** | **2.6162** | **0.7119** | 🏆 **Transformer (0.712%)** |
| **During Quiescent Dry Period** | 5,799 | **1.0341** | 2.7893 | 2.5590 | 4.3496 | 🏆 **Persistence (1.034%)** |
| **Overall Validation Split** | 6,326 | 1.2502 | 2.7630 | 2.5637 | 4.0465 | 🏆 **Persistence (Pooled)** |

#### Key Insight
1. **In the Active Irrigation Regime ($N = 527$):**
   - Persistence error explodes to **$3.629\%$** ($\text{RMSE} = 4.462\%$, $R^2 = -0.150$). Persistence fails because soil moisture increases substantially following irrigation.
   - **All three neural networks outperform persistence.**
   - The **Transformer** achieves an extraordinary **$0.7119\%$ MAE** ($\text{RMSE} = 1.243\%$, $R^2 = 0.9107$), beating the persistence baseline by **$5\times$**!
2. **In the Quiescent Dry Regime ($N = 5,799$ in Val, $N = 6,348$ in Test):**
   - Soil moisture is quasi-stationary. Persistence achieves $1.034\%$ (Val) and $0.412\%$ (Test).
   - Neural models produce small oscillatory predictions based on temperature and wind lags, yielding higher errors ($2.5\% - 4.3\%$).

### 5.3 Feasibility of a Defensible Event-Aware Split
Can a reliable, event-aware evaluation split be created without data leakage?

#### Feasible Proposed Split Scheme (Blocked Temporal Holdout)
Instead of a naive uniform percentage split (which dumps all models into the inactive September tail), a defensible protocol must use **Blocked Temporal Holdouts** during the active July–August window:
- **Sector 4 (Zucchini):**
  - Train: March 1 to July 20, 2025 (captures vegetative development + early irrigation).
  - Validation: July 21 to August 5, 2025 (active irrigation, $15$ days).
  - Test Holdout: August 6 to August 25, 2025 ($20$ days of contiguous, active irrigation: $\sim 2,800$ contiguous windows with $> 60\%$ active irrigation).
- **Sector 5 (Blueberry):**
  - Train: February 11 to July 25, 2025.
  - Validation: July 26 to August 10, 2025.
  - Test Holdout: August 11 to August 26, 2025 (contiguous active irrigation window).
- **Sectors 1 & 2:** Cannot support event-aware holdouts because sensor reporting was too sparse prior to July, and irrigation ceased by mid-July.

#### Additional Real-World Data Required
To evaluate irrigation forecasting with full scientific rigor:
1. Multi-season records spanning full crop cycles (planting to harvest) with synchronized 10-minute sensor telemetry.
2. Direct flow-meter telemetry (liters delivered per pulse) rather than binary valve timers.
3. Soil water tension sensors (tensiometers / watermark sensors in kPa) alongside capacitance probes to measure true soil matric potential.

---

## 6. Task 5: Production Compatibility & Sensor Scale Mismatch

### 6.1 Soil-Moisture Scale Decoupling

| Dimension | Italian Field Trial Dataset (`SUT*`) | Live ESP32 Hardware (GPIO 32, GPIO 33) |
| :--- | :--- | :--- |
| **Physical Sensor** | Agricultural commercial capacitance probes installed in root zone ($0–18\text{ cm}$). | Low-cost hobby capacitive sensors (v1.2). |
| **Output Quantity** | Permittivity calibrated against agronomic soil limits. | 12-bit raw analog-to-digital counts ($0–4095\text{ ADC}$). |
| **Calibration Basis** | Soil-specific water retention curve (PWP, MADP, FC, SP). | 2-point air/water index: dry air ($0\%$), submerged water ($100\%$). |
| **Field Capacity (FC)** | Defined at $60.0\%–90.0\%$ scale depending on crop. | Unknown (typically $25\%–45\%$ on an air/water scale in real soil). |
| **Wilting Point (PWP)**| Defined at $20.0\%–40.0\%$ scale. | Unknown (often $< 15\%$ on an air/water scale). |

### 6.2 Agronomic Risk & Safety Mandate
- Mapping the live ESP32 capacitive reading into the dataset's scale without **in-situ gravimetric soil calibration** would cause catastrophic decision errors. A reading of $40\%$ on the ESP32 could represent saturated potting mix, while the model interprets $40\%$ as severe drought stress.
- **Enforcement:**
  - `PUMP_ALLOWED = False` remains strictly hard-coded in all modules.
  - Dashboard predictions from the production 11-feature model (`random_forest_11feat.pkl`) are strictly **advisory**.
  - No automatic actuation can be deployed until site-specific gravimetric calibration curves are obtained.

---

## 7. Limitations & Prioritized Recommendations

### Limitations Identified
1. **Test Split Bias:** The existing test partition evaluates zero irrigation events, creating a misleading evaluation that penalizes dynamic sequence models.
2. **Circular Label Heuristics:** Ground-truth valve activations were filtered using the very moisture variable the models aim to predict.
3. **Sector Transferability:** Soil moisture levels vary from $2\%$ to $57\%$ across sectors due to different sensor placements and soil types, causing extreme ecological fallacies in pooled metrics.

### Prioritized Recommendations for Future Work
1. **Preserve Phase 11 Production Classifier as Sole Active Model:**  
   Continue using `random_forest_11feat.pkl` (threshold $0.3271$) in the dashboard for operational advisory monitoring.
2. **Adopt Regime-Stratified Reporting for Sequential Forecasting:**  
   When publishing or documenting experimental forecasting, report performance partitioned by regime (Quiescent vs Active Irrigation). Acknowledge that the Transformer excels during irrigation ($0.712\%$ MAE), while persistence excels during post-season rest ($0.412\%$ MAE).
3. **Do Not Alter Production Preprocessing or Calibrations:**  
   Keep `database.py` calibration offsets and `training/inference.py` feature mappings unchanged.
4. **Maintain Pump Actuation Lock:**  
   Keep `PUMP_ALLOWED = False` permanently locked until field gravimetric calibration is performed.

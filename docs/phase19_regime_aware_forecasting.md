# Smart Agriculture — Phase 19: Regime-Aware Forecasting Prototype (No Training)

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 19 — Regime-Aware Forecasting Prototype  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model, Threshold: `0.3271`)  
**Hardware Status:** ESP32 Dual Capacitive Probes (GPIO 32, GPIO 33), DHT22 — `PUMP_ALLOWED = False` Strictly Enforced  

---

## 1. Executive Summary & Research Motivation

Phases 17 and 18 revealed a fundamental dichotomy in soil-moisture dynamics:
- **During Quiescent Dry Equilibria:** Soil moisture changes by fractions of a percent over 24 hours. The naive persistence heuristic ($\hat{y} = x_t$) is near-optimal ($\text{MAE} = 0.412\%$ on the test set, $1.034\%$ on the validation set), while deep neural networks produce spurious fluctuations ($\text{MAE} = 3.867\% - 6.124\%$).
- **During Active Irrigation / Fertigation:** Soil moisture undergoes sharp upward pulses. Persistence fails severely ($\text{MAE} = 3.629\%$, $R^2 = -0.150$), while the experimental Transformer accurately captures the wetting response, achieving an $\text{MAE}$ of **$0.712\%$** ($R^2 = 0.9107$).

### Phase 19 Objective
Develop and evaluate a **regime-aware forecasting prototype** that combines the strengths of both paradigms:
1. Routing to the **Transformer** when an active irrigation/wetting regime is detected.
2. Defaulting to **Persistence** during quiescent periods.

**Scope Constraint:** Strictly non-destructive research evaluation. No models were trained or retrained. No production code, weights, or dashboard paths were modified.

---

## 2. Task 1: Audit of Available Forecasting Predictions & Split Integrity

All evaluations strictly reuse the existing saved artifacts:
- **Model Checkpoints:** [`transformer_experimental.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase15_transformer/transformer_experimental.pt), [`lstm_experimental.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/lstm_experimental.pt), [`tcn_experimental.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/tcn_experimental.pt).
- **Scalers:** 14-feature training mean and scale vectors fitted exclusively on the 70% training split.
- **Data Partitions:** Exact, un-altered chronological splits:
  - Validation Split: $N = 6,326$ contiguous 10-minute windows across Sectors 1, 2, 4, and 5 ($527$ event windows, $5,799$ quiescent windows).
  - Test Split: $N = 6,348$ contiguous 10-minute windows ($0$ event windows, $6,348$ quiescent windows).

---

## 3. Task 2: Regime Selector Design & Causal Gating Analysis

### 3.1 The Causal Information Boundary
In an operational deployment, a forecasting model cannot use future water volumes (`water_vol_to_24h`), future soil moisture, or retrospective event tags. The regime selector must decide strictly using inputs available at forecast origin $t$.

### 3.2 Training-Derived Causal Selector Rule
Analysis of the 70% training split ($N = 14,208$ windows) revealed that historical valve telemetry is an exceptionally clean causal indicator of an active wetting regime:
$$\text{Rule: } \text{past\_water\_4h} = \sum_{k=0}^{23} \text{irrigation\_duration\_minutes}[t-k] > 0$$

#### Training Split Performance:
- When $\text{past\_water\_4h} > 0$, the probability that the 24-hour forward window experiences future water is **$94.5\%$** (Precision = $94.5\%$).

#### Validation Split Gate Accuracy ($N = 6,326$):
- **True Positives (TP):** $75$ windows
- **False Positives (FP):** $25$ windows
- **False Negatives (FN):** $452$ windows
- **True Negatives (TN):** $5,774$ windows
- **Precision:** $\mathbf{75.0\%}$
- **Specificity:** $\mathbf{99.57\%}$
- **Recall:** $14.23\%$

### 3.3 The Fundamental Causal Limitation of Pre-Event Gating
Why is recall $14.23\%$?
- `past_water_4h > 0` triggers immediately once a valve has opened within the past 4 hours. It provides high precision for forecasting the ongoing 24-hour wetting front and percolation plateau.
- However, for windows occurring **5 to 20 hours before a valve opens**, no water has been applied yet.
- In commercial precision agriculture, fertigation cycles are scheduled by the farm manager or timer (e.g. daily at 14:00). Soil moisture before the event is above the wilting threshold ($81\%$ in Sector 4).
- **Causal Conclusion:** A sensor-only model looking at past soil moisture cannot foresee a future timer event without access to the farm automation calendar. In a real-world integrated system, this is solved by passing the **irrigation control schedule** into the forecasting engine as a planned input.

---

## 4. Task 3: Prototype Evaluation Across Slices & Sectors

Four strategies were compared using identical forecast origins and targets across all splits:
1. **Persistence Alone:** $\hat{y}_t = \text{soil\_moisture}[t]$
2. **Transformer Alone:** $\hat{y}_t = f_{\text{transformer}}(X_{t-23:t})$
3. **Causal Online Prototype:** Causal gate using $\text{past\_water\_4h} > 0 \to \text{Transformer}$; else $\text{Persistence}$.
4. **Retrospective Oracle:** Ideal theoretical upper bound routing to Transformer during retrospective event windows and Persistence during quiescent windows.

### 4.1 Overall Validation Split Benchmark ($N = 6,326$)

| Strategy | Overall MAE (%) | Overall RMSE (%) | Pooled $R^2$ | Event MAE (%) ($N=527$) | Quiescent MAE (%) ($N=5,799$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Alone** | 1.2502 | 2.3667 | 0.9901 | 3.6287 | 1.0341 |
| **Transformer Alone** | 4.0465 | 5.9812 | 0.9371 | **0.7119** | 4.3496 |
| **Causal Online Prototype** | **1.2048** | **2.2838** | **0.9908** | 3.2349 | **1.0203** |
| **Retrospective Oracle** | **1.0073** | **2.0178** | **0.9928** | **0.7119** | 1.0341 |

#### Core Performance Insights:
1. **The Causal Online Prototype Beats Persistence Overall:**  
   Overall validation $\text{MAE}$ improves from **$1.2502\%$** to **$1.2048\%$**, and $\text{RMSE}$ drops from **$2.3667\%$** to **$2.2838\%$**. By switching to the Transformer when active valve operation is detected, the causal gate improves performance on active wetting rows while safely defaulting to Persistence during dry periods.
2. **Retrospective Oracle Shows the Full Potential:**  
   If scheduled events are known in advance, routing drops overall error to **$1.0073\%$** ($\text{RMSE} = 2.0178\%$).

### 4.2 Test Split Verification ($N = 6,348$, 100% Quiescent Post-Season)

| Strategy | Overall Test MAE (%) | Overall Test RMSE (%) | Test $R^2$ | Behavioral Audit |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence Alone** | **0.4124** | **0.7641** | **0.9986** | Near-optimal under static post-season equilibrium. |
| **Transformer Alone** | 6.1237 | 8.7057 | 0.8222 | Catastrophic failure; over-predicts non-existent oscillations. |
| **Causal Online Prototype** | **0.4124** | **0.7641** | **0.9986** | **100% shielded.** Correctly detects zero past water and routes to Persistence. |
| **Retrospective Oracle** | **0.4124** | **0.7641** | **0.9986** | Identical to Persistence. |

*Shielding Proof:* The causal regime gate completely prevents the $6.12\%$ error of the un-gated Transformer on the test set.

### 4.3 Sector-Level Validation Performance Breakdown

| Sector | Metric | Persistence | Transformer | Causal Online Prototype | Retrospective Oracle | Event Windows ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sector 1 (Tomato)** | MAE (%) | **0.4092** | 0.4489 | **0.4092** | **0.4092** | 0 / 1,383 |
| **Sector 2 (Tomato)** | MAE (%) | **0.2616** | 1.5809 | **0.2616** | **0.2616** | 0 / 1,743 |
| **Sector 4 (Zucchini)**| MAE (%) | 1.6609 | 7.9802 | **1.5020** | **0.9301** | 477 / 1,809 |
| **Sector 5 (Blueberry)**| MAE (%) | **2.7912** | 5.5973 | **2.7912** | **2.6366** | 50 / 1,391 |

In Sector 4, where the bulk of active irrigation occurred, the Causal Online Prototype reduces $\text{MAE}$ from **$1.6609\%$** down to **$1.5020\%$** (and $\text{RMSE}$ from $2.7914\%$ to $2.5385\%$).

---

## 5. Task 4: Limitations & Generalizability Assessment

1. **Extreme Sector Concentration:**
   $90.5\%$ of event windows ($477 / 527$) originate from Sector 4 (Zucchini). Sector 5 contributes $9.5\%$, while Sectors 1, 2, and 3 have zero event windows. The prototype has not been evaluated across diverse crop systems during active irrigation.
2. **Small Number of Independent Events:**
   The $527$ validation event windows represent rolling views over **only 8 fertigation pulses** ($15$-minute duration) across **6 calendar-day blocks**.
3. **Temporal Overlap:**
   Adjacent windows share $95.8\%$ of input features and $99.3\%$ of target steps. High autocorrelation means that the effective sample size is closer to $N \approx 6 - 8$ independent event episodes.
4. **Lack of Independent Event-Rich Test Data:**
   In the standard chronological split, all earlier active irrigation (February–August) was consumed by the training set, while the test set has zero irrigation. No additional unseen event-rich data exists in this dataset without modifying split boundaries and retraining.
5. **Generalizability Finding:**
   The prototype cannot be claimed to generalize to unseen growing seasons or open-field flood irrigation without dedicated multi-season verification.

---

## 6. Task 5: Production Isolation & Safety Verification

### 6.1 Architectural Isolation
- Prototype module created: [`training/experimental/phase19/regime_prototype.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase19/regime_prototype.py).
- The module is strictly isolated under `training/experimental/phase19/`.
- No imports or dependencies connect `regime_prototype.py` to [`training/inference.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/inference.py) or [`dashboard.py`](file:///Users/aryanagrawal/ESA_PROJECT/dashboard.py).

### 6.2 Hardware Safety Lock
- `PUMP_ALLOWED = False` is explicitly declared and asserted in `regime_prototype.py`.
- No GPIO relay actuation, pump commands, or external actuator signals exist.

### 6.3 Active Production Classifier
- Production model: `training/models/random_forest_11feat.pkl` (Phase 11 HistGradientBoosting, tuned threshold `0.3271`).
- Feature schema: Exactly 11 environmental features.
- Dashboard operational status: Sole active model, advisory only.

---

## 7. Deliverables Summary

1. **Python Prototype Module:** [`training/experimental/phase19/regime_prototype.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase19/regime_prototype.py)  
   Implements `causal_regime_gate()` and `predict_regime_aware()` with hard-locked safety assertions.
2. **Results JSON:** [`training/experimental/phase19/phase19_results.json`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase19/phase19_results.json)  
   Full quantitative record of all validation and test split evaluations across persistence, deep learning, causal gating, and oracle strategies.
3. **Markdown Documentation:** [`docs/phase19_regime_aware_forecasting.md`](file:///Users/aryanagrawal/ESA_PROJECT/docs/phase19_regime_aware_forecasting.md)  
   Comprehensive audit of regime selection, causal gating feasibility, and statistical limitations.

---

## 8. Prioritized Recommendations for Future Work

1. **Maintain Production Isolation:** Keep `random_forest_11feat.pkl` as the sole operational advisory classifier in the dashboard.
2. **Preserve Pump Safety Lock:** Keep `PUMP_ALLOWED = False` locked.
3. **Integrate Control-Policy Signals for Future Forecasting:** Rather than expecting passive sensor data to predict when a human operator will turn on a valve, future research should explore **schedule-conditioned forecasting**, where planned irrigation events are provided as an explicit schedule input.

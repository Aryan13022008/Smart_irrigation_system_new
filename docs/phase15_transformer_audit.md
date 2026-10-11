# Smart Agriculture — Phase 15: Isolated Experimental Transformer Report

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 15 — Isolated Experimental Transformer for Soil-Moisture Forecasting  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model, Threshold: 0.3271)  
**Hardware Status:** ESP32 (GPIO 32, GPIO 33 capacitive probes, DHT22) — Automatic Pump Actuation STRICTLY DISABLED  

---

## Executive Summary & Architecture Verdict

A compact, regularized PyTorch Transformer encoder was designed, trained, and evaluated in an isolated research environment ([`training/experimental/phase15_transformer/`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase15_transformer/)) to forecast 24-hour mean soil moisture (`target_mean_24h`) from 24 contiguous 10-minute historical observations of 14 features.

### Empirical Verdict: 🔴 NO-GO FOR PRODUCTION DEPLOYMENT
- The **naive persistence baseline** ($\hat{y} = \text{soil\_moisture}[t]$) achieved a test MAE of **$0.4124\%$**.
- The **Transformer** achieved a test MAE of **$6.1237\%$** (nearly **$15\times$ higher error** than the baseline).
- The Transformer also lagged behind both the experimental LSTM ($3.8665\%$) and TCN ($4.6904\%$).
- **Conclusion:** In accordance with rigorous empirical standards, the Transformer is **not deployed to production** and is documented purely as an offline research artifact.

---

## 1. Transformer Architecture & Training Configuration

The model was intentionally designed with modest capacity to prevent severe overfitting:

| Parameter | Specification | Rationale / Detail |
|:---|:---|:---|
| **Architecture** | PyTorch Transformer Encoder | Sequence-to-scalar regression with residual attention. |
| **Input Shape** | $(B, 24, 14)$ | 24 contiguous 10-minute timesteps (4 hours) $\times$ 14 features. |
| **Embedding Dimension ($d_{\text{model}}$)** | 32 | Linear projection from 14 raw features to 32 dimensions. |
| **Positional Encoding** | Sinusoidal ($d_{\text{model}} = 32$) | Encodes temporal ordering across the 24 timesteps. |
| **Attention Heads ($n_{\text{heads}}$)** | 4 | 8 dimensions per head; modest multi-head capacity. |
| **Encoder Layers ($n_{\text{layers}}$)** | 2 | Shallow depth to prevent over-parameterization. |
| **Feedforward Dimension ($d_{\text{ff}}$)** | 64 | $2\times d_{\text{model}}$ expansion ratio. |
| **Dropout** | 0.2 | Applied across embedding, self-attention, and MLP head. |
| **Regression Head** | $\text{Linear}(32 \to 16) \to \text{ReLU} \to \text{Dropout} \to \text{Linear}(16 \to 1)$ | Maps last timestep token representation to predicted moisture. |
| **Total Parameters** | **18,113** | Comparable to TCN (17,921) and LSTM (6,689). |
| **Loss Function** | Mean Squared Error ($\text{MSE}$) | Standard regression objective on $\text{target\_mean\_24h}$. |
| **Optimizer** | AdamW ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-4}$) | Coupled with `ReduceLROnPlateau` scheduler. |
| **Early Stopping** | Patience 8 epochs on validation loss | Triggered at epoch 14 (Best Val MSE: 35.9159 at epoch 6). |

---

## 2. Target Construction Audit & Inconsistency Explanation

### 2.1 Target Definition & Alignment
In [`Soil Moisture, Irrigation Actuator and Weather Dat/03_code/02_preprocessing.py`](file:///Users/aryanagrawal/ESA_PROJECT/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/03_code/02_preprocessing.py):
- `target_mean_24h` is calculated as the forward rolling mean over 144 steps of 10 minutes ($[t, t+143]$) using `FixedForwardWindowIndexer(window_size=144)`.
- The current timestep $t$ is included in the window, meaning $\text{target\_mean\_24h}$ averages the subsequent 24 hours starting from the observation point.

### 2.2 Why the Persistence Baseline Achieves MAE 0.4124 vs Neural Nets > 3.8
1. **Stationary Post-Season Test Set:**
   - The chronological test split encompasses the final 15% of records for each sector (late August through late September 2025).
   - In this post-season testing period, **zero irrigation events occurred across all sectors** ($0$ out of $6,348$ windows had $\text{water\_vol\_to\_24h} > 0$).
   - Because no artificial watering occurred, soil moisture was in natural equilibrium with minimal fluctuations (standard deviation in Sector 1 was just $0.16\%$).
   - Consequently, the current soil moisture at step $t$ was almost mathematically identical to the 24-hour mean, giving the persistence baseline a tiny MAE of **$0.4124\%$**.
2. **Dynamic Overfitting in Neural Networks:**
   - The neural networks (LSTM, TCN, and Transformer) were trained on the active growing season (February–August), during which frequent irrigation events caused large moisture shifts.
   - The neural networks learned dynamic response functions linking weather variations to moisture changes.
   - When evaluated on the static post-season test set, their dynamic predictions fluctuated around the stationary baseline, resulting in substantially higher errors ($3.867\%$ for LSTM, $4.690\%$ for TCN, and $6.124\%$ for Transformer).
3. **Simpson's Paradox in Pooled $R^2$:**
   - In the pooled test set, variance across sectors is massive ($Var(y) \approx 780$) because Sector 5 operates near $2\%$, Sectors 1 & 2 operate near $15–20\%$, and Sector 4 operates near $57\%$.
   - Any model that roughly separates the sectors achieves an artificially high global $R^2$ ($> 0.82$).
   - Within individual sectors, however, the target variance is tiny ($0.025$ in Sector 1), causing within-sector $R^2$ to collapse to negative values for all neural network architectures.

---

## 3. Four-Way Model Benchmark on Identical Test Windows ($N = 6,348$)

All models were evaluated on the exact same $6,348$ contiguous 10-minute held-out test windows across Sectors 1, 2, 4, and 5:

### 3.1 Overall Metrics Benchmark

| Model | Parameters | Test MAE (%) | Test RMSE (%) | Pooled Test $R^2$ | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Persistence Baseline (Current Moisture)** | **0** | **0.4124** | **0.7641** | **0.99863** | **Authoritative Benchmark** |
| **Experimental LSTM** | 6,689 | 3.8665 | 5.1434 | 0.93794 | Offline Research Artifact |
| **Experimental TCN** | 17,921 | 4.6904 | 7.6977 | 0.86098 | Offline Research Artifact |
| **Experimental Transformer** | 18,113 | 6.1237 | 8.7057 | 0.82219 | Offline Research Artifact |

### 3.2 Per-Zone Metrics Breakdown

| Sector | Crop & System | Test Samples | Target Mean ± Std | Persistence MAE (%) | Transformer MAE (%) | LSTM MAE (%) | TCN MAE (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sector 1** | Tomato, Open Field | 1,408 | 19.26% ± 0.16% | **0.4678** | 0.8398 | 1.4221 | 2.2372 |
| **Sector 2** | Tomato, Open Field | 1,721 | 15.53% ± 0.26% | **0.3810** | 1.9944 | 2.6545 | 2.1904 |
| **Sector 3** | Tomato, Pots | 0 | — | N/A | N/A | N/A | N/A |
| **Sector 4** | Zucchini | 1,599 | 57.27% ± 1.89% | **0.3807** | 5.4820 | 6.2999 | 1.6669 |
| **Sector 5** | Blueberry | 1,620 | 2.20% ± 1.97% | **0.4290** | 15.7363 | 4.8766 | 12.4626 |

**Key Finding:** The persistence baseline is superior to all three deep learning sequence models across every single sector. The Transformer exhibited particularly poor performance on Sector 5 (MAE $15.74\%$), where attention weights over-predicted wetting dynamics in flatlined soil.

---

## 4. Preserved Production System & Safety Locks

1. **Production Model Untouched:**
   - File: [`training/models/random_forest_11feat.pkl`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/random_forest_11feat.pkl)
   - Class: `HistGradientBoostingClassifier`
   - Decision threshold: `0.3270985223048995`
   - Role: Sole active model for irrigation decision recommendations on the dashboard.
2. **Pump Control Hard-Locked:**
   - In all forecasting modules ([`forecasting_models.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase13/forecasting_models.py) and [`transformer_inference.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/experimental/phase15_transformer/transformer_inference.py)), `PUMP_ALLOWED = False` is enforced at the module level.
   - Automatic pump relay actuation remains strictly disabled across the entire codebase.

---

## 5. Automated Test Suite Verification

All 45 tests across the three test suites were executed:

```bash
python3 -m unittest test_ml_integration.py test_phase14_integration.py test_phase15_transformer.py
```

### Test Results Summary:
- **`test_ml_integration.py` (25 tests):** **PASS (100%)** — Asserts Phase 11 production 11-feature model display, threshold $0.3271$, zero model dropdowns, and dashboard AppTest rendering.
- **`test_phase14_integration.py` (10 tests):** **PASS (100%)** — Asserts LSTM/TCN loading, 14-feature ordering, chronological split integrity, baseline recalculation, and zone isolation.
- **`test_phase15_transformer.py` (10 tests):** **PASS (100%)** — Asserts Transformer checkpoint loading, shape validation, missing-data rejection, sequence continuity, baseline dominance, and pump safety lock.
- **Total:** **45 / 45 PASS (100%)** in 7.4 seconds.

---

## 6. How to Reproduce & Run Experiments

```bash
# 1. Run Transformer training and evaluation:
python3 training/experimental/phase15_transformer/train_transformer_forecasting.py

# 2. Run the Phase 15 test suite:
python3 -m unittest test_phase15_transformer.py

# 3. Run all project tests:
python3 -m unittest test_ml_integration.py test_phase14_integration.py test_phase15_transformer.py

# 4. View in Dashboard:
streamlit run dashboard.py
```
*(In dashboard: Navigate to **🤖 AI Models** $\to$ inspect **🧪 Experimental 24-Hour Soil-Moisture Forecasting (Research)**).*

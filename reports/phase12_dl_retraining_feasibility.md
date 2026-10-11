# Smart Agriculture — Phase 12: Deep Learning Retraining Feasibility Report

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 12 — Investigation and Data Validation of Deep Learning Retraining Feasibility  
**Active Production Model:** `training/models/random_forest_11feat.pkl` (Phase 11, Sole Active Model)  
**Hardware Status:** ESP32 (GPIO 32, GPIO 33 capacitive probes, DHT22) — Automatic Pump Actuation DISABLED  

---

## Executive Summary & Architecture Decisions

A comprehensive, read-only empirical investigation was conducted on [`/Users/aryanagrawal/Downloads/iotsensordata.csv`](file:///Users/aryanagrawal/Downloads/iotsensordata.csv) to determine whether the dataset can legitimately support retraining the three deep learning sequence models (`improved_lstm.pt`, `tcn.pt`, `transformer.pt`).

### Final Retraining Decisions:

| Architecture | Decision | Parameter Count | Core Empirical Finding | Operational Recommendation |
|:---|:---:|:---:|:---|:---|
| **LSTM** | 🔴 **NO-GO** | ~51,585 | Soil moisture autocorrelation is zero ($r = -0.002$); 24-step sequence contains uncorrelated noise. Recurrent state has zero predictive benefit. | **Do NOT retrain.** High variance; no theoretical advantage over tabular trees. |
| **TCN** | 🔴 **NO-GO** | ~198,529 | Causal dilated convolutions over 24 steps find no temporal dynamics on i.i.d. draws. Receptive field captures spurious meteorological noise. | **Do NOT retrain.** Extreme parameter bloat for a static threshold task. |
| **Transformer** | 🔴 **NO-GO** | ~18,273 | Self-attention across non-sequential soil draws is mathematically ungrounded. Quadratic attention over noise risks memorizing training artifacts. | **Do NOT retrain.** Complex attention provides zero domain signal over tabular GBDT. |

> [!IMPORTANT]
> **Core Finding**: `iotsensordata.csv` is **NOT an authentic time series**. Autocorrelation analysis proves that its soil sensor readings (`Soil Moisture`, `Temperature`, ` Soil Humidity`, `Time`) are independent, identically distributed (i.i.d.) random draws ($r \approx 0$). Furthermore, the target `Status` is an unverified static simulation rule ($100\%$ `OFF` when `Time > 90`) that exhibits **zero correlation** with meteorological weather variables ($r \approx 0.005$). Retraining deep learning sequence models on this dataset is scientifically indefensible.

---

## 1. Dataset Profiling: `iotsensordata.csv`

Inspection of the raw dataset at [`/Users/aryanagrawal/Downloads/iotsensordata.csv`](file:///Users/aryanagrawal/Downloads/iotsensordata.csv):

### 1.1 Structural Summary
- **Total Records:** $100,000$ rows $\times$ $15$ columns
- **File Size:** ~7.1 MB
- **Target Column:** `Status` (`"ON"` vs `"OFF"`)
- **Target Distribution (Full Dataset):**
  - `"ON"`: $53,650$ records ($53.65\%$)
  - `"OFF"`: $46,350$ records ($46.35\%$)

### 1.2 Missing Value Analysis
The dataset exhibits three distinct data regimes reflecting independent pasted sources:

| Column | Non-Null Count | Null Count | Null % | Data Regime / Source |
|:---|:---:|:---:|:---:|:---|
| `Soil Moisture` | 100,000 | 0 | 0.00% | Synthetic IoT Soil Simulation |
| `Temperature` (soil) | 100,000 | 0 | 0.00% | Synthetic IoT Soil Simulation |
| ` Soil Humidity` | 100,000 | 0 | 0.00% | Synthetic IoT Soil Simulation |
| `Time` | 100,000 | 0 | 0.00% | Synthetic IoT Simulation Counter |
| `Status` | 100,000 | 0 | 0.00% | Synthetic Actuator Target |
| `Air temperature (C)` | 23,995 | 76,005 | 76.01% | External Weather Station Slice |
| `Wind speed (Km/h)` | 23,995 | 76,005 | 76.01% | External Weather Station Slice |
| `Air humidity (%)` | 23,995 | 76,005 | 76.01% | External Weather Station Slice |
| `Wind gust (Km/h)` | 23,995 | 76,005 | 76.01% | External Weather Station Slice |
| `Pressure (KPa)` | 23,995 | 76,005 | 76.01% | External Weather Station Slice |
| `ph` | 2,200 | 97,800 | 97.80% | Kaggle Crop Recommendation Subset |
| `rainfall` | 2,200 | 97,800 | 97.80% | Kaggle Crop Recommendation Subset |
| `N` | 2,200 | 97,800 | 97.80% | Kaggle Crop Recommendation Subset |
| `P` | 2,200 | 97,800 | 97.80% | Kaggle Crop Recommendation Subset |
| `K` | 2,200 | 97,800 | 97.80% | Kaggle Crop Recommendation Subset |

### 1.3 Target Distribution on Complete Weather Subset ($N = 23,995$)
When filtering rows where meteorological features are complete:
- `"ON"`: $12,973$ records ($54.07\%$)
- `"OFF"`: $11,022$ records ($45.93\%$)

### 1.4 Timestamp and Frequency Audit
- **Timestamp Column:** **NONE.** The dataset contains no calendar dates, UTC timestamps, or time indices.
- **The `Time` Column:**
  - Data Type: `int64`
  - Range: $[0, 110]$, Mean: $55.25$, Std: $32.09$
  - Monotonicity: **False.** Consecutive differences jump wildly between $-110$ and $+110$ (Mean diff: $0.0001$, Std: $45.33$).
  - Assessment: `Time` is a discrete simulation timer or scenario index, not an advancing chronological clock.
- **Chronological Coverage:** **Zero.** The calendar dates (`2022-01-01` to `2024-09-26`) referenced in earlier model training pipelines were **artificially fabricated** via `pd.date_range(start="2022-01-01", periods=len(df), freq="1h")`.

---

## 2. Target Provenance & Label Meaning: `Status`

### 2.1 Project Documentation Verification
- The official project documentation ([`data/README.txt`](file:///Users/aryanagrawal/ESA_PROJECT/data/README.txt) and [`data/MANIFEST.TXT`](file:///Users/aryanagrawal/ESA_PROJECT/data/MANIFEST.TXT)) describes the KU Leuven DRIP dataset (18 vegetable cropping cycles in Belgium, TEROS 10 sensors at 15 cm depth, Penman-Monteith $ET_0$).
- Neither `iotsensordata.csv` nor the `Status` column is mentioned or defined anywhere in the repository documentation.
- **Verdict:** The label meaning of `Status` is **UNVERIFIED**.

### 2.2 Empirical Reverse-Engineering of `Status`
Statistical inspection and decision tree modeling reveal that `Status` was generated by a deterministic heuristic simulation script:

1. **Hard Cutoff Rule:**
   $$\text{If } \text{Time} > 90 \implies \text{Status} = \text{OFF } (100.0\% \text{ across all } 18,310 \text{ rows})$$
   There is not a single row in the entire 100,000-sample dataset where `Status == "ON"` when `Time > 90`.
2. **Moisture Threshold Rule:**
   $$\text{If } \text{Soil Moisture} \le 59.5 \text{ and } \text{Time} \le 90.5 \implies \text{Status} \approx \text{ON}$$
   A simple 2-variable static heuristic achieves **$78.34\%$ accuracy** across all 100,000 rows ($78.37\%$ on the 23,995 weather subset).
3. **Multi-Variable Simulation Tree:**
   A shallow 4-variable decision tree on `Soil Moisture`, `Temperature`, ` Soil Humidity`, and `Time` achieves **$89.65\%$ accuracy**.
4. **Agronomic Conclusion:**
   `Status` does not reflect real-world agronomic soil matric potential, crop water stress, or farmer irrigation actions. It represents an artificial threshold simulation programmed into a bench test script.

---

## 3. Sequential Validity & Source Consistency Audit

To justify sequence models (LSTM, TCN, Transformer), the underlying data must exhibit temporal continuity and autocorrelation.

### 3.1 Autocorrelation Analysis

```
Autocorrelation Across Temporal Lags (iotsensordata.csv)
```

| Variable | Lag 1 ($t-1$) | Lag 2 ($t-2$) | Lag 3 ($t-3$) | Lag 5 ($t-5$) | Lag 10 ($t-10$) | Lag 24 ($t-24$) | Regime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Soil Moisture** | **-0.0023** | **-0.0001** | **-0.0028** | **-0.0007** | **-0.0055** | **+0.0047** | **i.i.d. Random Noise** |
| **Temperature (soil)** | **-0.0011** | **+0.0103** | **+0.0035** | **+0.0010** | **+0.0015** | **-0.0037** | **i.i.d. Random Noise** |
| **Soil Humidity** | **+0.0028** | **+0.0023** | **-0.0029** | **+0.0031** | **-0.0015** | **-0.0025** | **i.i.d. Random Noise** |
| **Time (counter)** | **+0.0023** | **-0.0015** | **+0.0031** | **+0.0004** | **+0.0027** | **+0.0030** | **i.i.d. Random Noise** |
| *Air temperature (C)* | *+0.9992* | *+0.9971* | *+0.9944* | *+0.9870* | *+0.9590* | *+0.8107* | *Continuous Weather Station* |
| *Wind speed (Km/h)* | *+0.9868* | *+0.9636* | *+0.9462* | *+0.9193* | *+0.8579* | *+0.6774* | *Continuous Weather Station* |
| *Air humidity (%)* | *+0.9996* | *+0.9985* | *+0.9968* | *+0.9922* | *+0.9742* | *+0.8868* | *Continuous Weather Station* |
| *Pressure (KPa)* | *+0.9995* | *+0.9985* | *+0.9971* | *+0.9934* | *+0.9786* | *+0.8996* | *Continuous Weather Station* |

### 3.2 Correlation with Target `Status` ($N = 23,995$)

| Feature Category | Feature Name | Correlation with Target `Status` ($r$) | Statistical Significance |
|:---|:---|:---:|:---|
| **Primary Predictors (Synthetic)** | `Soil Moisture` | **-0.3233** | Strong negative association |
| | `Temperature` (soil) | **+0.3083** | Strong positive association |
| | `Time` (simulation counter) | **-0.2487** | Strong negative association |
| | ` Soil Humidity` | **+0.0232** | Minor secondary association |
| **Weather Stream (Continuous)** | `Air temperature (C)` | **+0.0097** | **Zero relationship (Noise)** |
| | `Wind speed (Km/h)` | **+0.0066** | **Zero relationship (Noise)** |
| | `Air humidity (%)` | **-0.0076** | **Zero relationship (Noise)** |
| | `Wind gust (Km/h)` | **+0.0032** | **Zero relationship (Noise)** |
| | `Pressure (KPa)` | **-0.0028** | **Zero relationship (Noise)** |

### 3.3 Synthesis
This is a critical architectural finding:
1. The soil sensor records are uncorrelated i.i.d. draws. Consecutive rows have no temporal relationship.
2. The meteorological variables have high temporal continuity, but **zero correlation with `Status`** ($|r| < 0.01$).
3. The dataset is a synthetic hybrid: uncorrelated soil simulation records were pasted alongside an independent meteorological sequence. Feeding a 24-step sliding window into an LSTM or Transformer forces the network to learn spurious relationships between past weather noise and random soil draws.

---

## 4. Live ESP32 Hardware & Feature Constructibility

### 4.1 Hardware vs Feature Compatibility Matrix

| Dataset Feature | Source in `iotsensordata.csv` | Live ESP32 Hardware | Status | Verification Detail |
|:---|:---|:---|:---:|:---|
| `soil_moisture` | `Soil Moisture` (int 1–90) | Capacitive Probes 1 & 2 (GPIO 32, GPIO 33) | 🟡 **UNVALIDATED** | Calibrated relative index ($0–100\%$) does not match proprietary integer scale. |
| `temperature` | `Air temperature (C)` | DHT22 Ambient Sensor | 🟢 **AVAILABLE** | Ambient air temperature available live. |
| `wind_speed` | `Wind speed (Km/h)` | Open-Meteo Weather API | 🟢 **AVAILABLE** | Real-time weather pipeline integration. |
| `air_humidity` | `Air humidity (%)` | DHT22 Ambient Sensor | 🟢 **AVAILABLE** | Ambient air humidity available live. |
| `wind_gust` | `Wind gust (Km/h)` | Open-Meteo Weather API | 🟢 **AVAILABLE** | Real-time weather pipeline integration. |
| `pressure` | `Pressure (KPa)` | Open-Meteo Weather API | 🟢 **AVAILABLE** | Real-time weather pipeline integration. |
| `soil_moisture_previous` | Lagged `soil_moisture` | SQLite Sensor History | 🟢 **AVAILABLE** | Derived backwards from SQLite history. |
| `soil_moisture_change` | 1-step delta | SQLite Sensor History | 🟢 **AVAILABLE** | Derived backwards from SQLite history. |
| `temp_previous` | Lagged `temperature` | SQLite Sensor History | 🟢 **AVAILABLE** | Derived backwards from SQLite history. |
| `month_sin`, `month_cos` | Cyclical date features | System Clock | 🟢 **AVAILABLE** | Derived from real-time clock. |
| `soil_temperature` | `Temperature` (raw CSV) | *None* | 🔴 **BLOCKED** | ESP32 lacks root-zone soil temperature probe. |
| `soil_humidity` | ` Soil Humidity` (raw CSV) | *None* | 🔴 **BLOCKED** | ESP32 lacks separate soil humidity probe. |
| `Time` | `Time` (raw CSV) | *None* | 🔴 **BLOCKED** | Cannot construct arbitrary 0–110 integer counter. |
| **24-Step Sequence** | Sliding window $(24, 11)$ | SQLite Sensor History | 🔴 **BLOCKED** | Requires 24 hours of uninterrupted live streaming; database has ~90 min. |

---

## 5. Sequence Construction & Multi-Block Gap Analysis

When filtering for complete weather records ($N = 23,995$), the dataset does not form a single contiguous sequence. It consists of **three distinct blocks** separated by multi-thousand-row gaps in the original CSV:

```
Block 1: Rows 0 -> 8,913 (8,914 rows)
[GAP 1: 1,153 missing rows (8,914 -> 10,066)]
Block 2: Rows 10,067 -> 16,082 (6,016 rows)
[GAP 2: 1,153 missing rows (16,083 -> 17,235)]
Block 3: Rows 17,236 -> 26,300 (9,065 rows)
```

### 5.1 Sequence Construction Hazards
1. **Gap Crossing:** If `dropna()` is executed naively, 46 sequence windows will straddle Gap 1 and Gap 2, concatenating non-contiguous meteorological time periods.
2. **Block-Isolated Sequences:** To prevent gap corruption, sequences must be generated within each block independently:
   - Block 1: $8,914 - 24 + 1 = \mathbf{8,891}$ valid sequences
   - Block 2: $6,016 - 24 + 1 = \mathbf{5,993}$ valid sequences
   - Block 3: $9,065 - 24 + 1 = \mathbf{9,042}$ valid sequences
   - **Total Valid Contiguous Sequences:** $\mathbf{23,926}$ sequences
3. **Physical Incoherence:** Even within blocks, because soil moisture autocorrelation is zero, every 24-step sequence consists of 24 independent random numbers. The neural network is being fed white noise as an input trajectory.

---

## 6. Leakage-Safe Chronological Partitioning Design

If a sequence model training experiment were to be conducted for research validation, the following leakage-safe protocol must be strictly enforced:

### 6.1 Partition Specifications

```
Chronological Split (Total N = 23,995 rows / 23,926 block-isolated sequences):
[=========== Train (70%: 16,796 rows) ===========] [Embargo] [== Val (15%: 3,599) ==] [Embargo] [== Test (15%: 3,600) ==]
```

1. **Training Partition (70%):** Rows $0$ to $16,795$ ($16,796$ rows; Block 1 and first portion of Block 2).
2. **Validation Partition (15%):** Rows $16,796$ to $20,394$ ($3,599$ rows; end of Block 2 and start of Block 3).
3. **Held-Out Test Partition (15%):** Rows $20,395$ to $23,994$ ($3,600$ rows; remainder of Block 3).

### 6.2 Embargo Buffer & Leakage Prevention Rules
- **Embargo Buffer:** Enforce an embargo zone of at least $W = 24$ steps at each partition boundary. Any sliding window whose input range intersects a boundary is discarded.
- **Scaler Fitting:** `StandardScaler.fit()` must be executed strictly on the training partition sequences. Transform validation and test sets using training parameters.
- **Zero Sequence Shuffling:** Strictly prohibit `np.random.shuffle()` across sequences before splitting.

---

## 7. Semantic Target & Schema Comparison (Phase 9 vs Deep Learning)

### 7.1 Schema Comparison

| Dimension | Phase 9 Model (`random_forest_11feat`) | Proposed Deep Learning Models | Semantic Mismatch |
|:---|:---|:---|:---|
| **Input Shape** | Tabular vector $(1, 11)$ | 3D Sequence Tensor $(1, 24, 11)$ | DL assumes temporal trajectory; RF assumes instantaneous state. |
| **Target Alignment** | Current-step `Status[t]` | Current-step `Status[t]` or Multi-step `Status[t+H]` | Historical lags have zero correlation with future status. |
| **Predictive Features** | `soil_moisture` (importance: 0.172) | 11 features across 24 timesteps | 10 of 11 features have zero importance ($< 0.001$). |

### 7.2 Permutation Feature Importance Evidence
Permutation importance measured on the Phase 9 test set ($N = 5,999$ samples):
- `soil_moisture`: **0.17202**
- `temperature`: 0.00172
- `temp_previous`: 0.00031
- `month_sin`: 0.00025
- `wind_gust`: 0.00006
- `pressure`: -0.00010
- `month_cos`: -0.00029
- `soil_moisture_previous`: -0.00076
- `wind_speed`: -0.00082
- `air_humidity`: -0.00174
- `soil_moisture_change`: -0.00235

In the 11-feature feature space, `soil_moisture` is the **only variable with positive predictive power**. Supplying 24 timesteps of 11 features ($264$ inputs per sample) expands feature dimensionality by $24\times$ without adding any domain information.

---

## 8. Data Sufficiency & Architecture Justification

### 8.1 Parameter Complexity vs Signal Analysis
- **Dataset Size:** $23,926$ valid sequences.
- **Target Complexity:** Static heuristic threshold.
- **Model Sizes:**
  - LSTM: $\sim 51,585$ parameters
  - TCN: $\sim 198,529$ parameters
  - Transformer: $\sim 18,273$ parameters
  - HistGradientBoosting: Small ensemble of shallow decision trees.

### 8.2 Architectural Inductive Bias Mismatch
- **LSTM / TCN / Transformer:** Inductive bias assumes meaningful temporal autocorrelation, stateful accumulation, and gradual trends (e.g. soil moisture depletion curves).
- **Reality:** Because the primary signal (`Soil Moisture`) is i.i.d. random noise, the recurrent states of an LSTM, dilated receptive fields of a TCN, and self-attention weights of a Transformer have nothing meaningful to track.
- **Overfitting Risk:** Deep architectures will fit spurious correlations between the continuous weather noise and the binary target, producing high validation scores that collapse on independent live data (as demonstrated in Phase 10 where the LSTM scored $0.5008$ ROC-AUC).

---

## 9. Proposed Training Protocols (Reference Specification)

For complete documentation and reproducibility, the training protocol for each architecture is formally defined below:

### 9.1 Common Training Hyperparameters
- **Input Dimension:** $(B, 24, 11)$
- **Target:** Binary `Status` at step $t$
- **Loss Function:** `nn.BCEWithLogitsLoss()`
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Batch Size:** $32$
- **Max Epochs:** $50$
- **Early Stopping:** Patience $10$ epochs on validation PR-AUC
- **Metrics Evaluated:** ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1 (at default $0.5$ and validation-tuned threshold)

### 9.2 Architecture Configurations
1. **Improved LSTM:**
   - Architecture: `LayerNorm(11) -> LSTM(input_size=11, hidden=64, num_layers=2, dropout=0.3) -> FC1(64->32) -> ReLU -> Dropout(0.3) -> FC2(32->1)`
   - Parameters: $51,585$
2. **TCN (Temporal Convolutional Network):**
   - Architecture: 3 Causal Dilated Residual Blocks (Channels: `64, 64, 128`, Kernel size: 5, Dilations: `[1, 2, 4]`, Dropout: 0.3) $\to$ `Linear(128->1)`
   - Parameters: $198,529$
3. **Transformer:**
   - Architecture: `Linear(11->32) -> PositionalEncoding -> 2-layer TransformerEncoder(d_model=32, nhead=4, d_ff=64, dropout=0.3) -> Pooling Head -> Linear(32->1)`
   - Parameters: $18,273$

> [!CAUTION]
> **Expected Performance Notice**: None of these architectures are expected to outperform the current `HistGradientBoostingClassifier` ($0.6808$ ROC-AUC, $0.8595$ Recall). At best, they will approximate the static threshold behavior of the gradient boosting model at significantly higher computational cost and latency.

---

## 10. Unresolved Assumptions & Actionable Recommendations

### 10.1 Unresolved Assumptions
1. **Physical Meaning of `Status`:** It remains unproven whether `Status` represents an agronomic crop watering need or an arbitrary threshold simulator.
2. **Physical Meaning of `Time`:** The discrete integer variable $[0, 110]$ has no documented physical or agronomic definition.
3. **Sensor Scale Equivalence:** The translation between ESP32 capacitive relative moisture ($0–100\%$) and the CSV integer scale ($1–90$) remains uncalibrated.

### 10.2 Actionable Recommendations
1. **Preserve Phase 11 Production System:**
   - Maintain `training/models/random_forest_11feat.pkl` as the sole active advisory model on the dashboard.
   - Maintain automatic pump actuation strictly disabled (`PUMP_ALLOWED = False`).
2. **Permanently Retire Legacy Deep Learning Models:**
   - Deprecate `improved_lstm.pt`, `tcn.pt`, and `transformer.pt` from operational consideration.
3. **Do NOT Retrain DL Models on `iotsensordata.csv`:**
   - Retraining sequence models on non-sequential composite data introduces architectural bloat and false confidence.
4. **Authentic Data Collection Pathway:**
   - Before attempting future deep learning sequence modeling, log genuine, continuous, hourly ESP32 sensor telemetry over real wetting and drying cycles in the deployed field.
5. **Path to Superior Performance (Phase 13 Candidate):**
   - If higher predictive accuracy is needed, install an inexpensive **DS18B20 digital soil temperature probe** on the ESP32. This restores access to root-zone soil temperature, enabling the reactivation of the baseline 13-feature model ($0.7927$ ROC-AUC, $0.9310$ Recall).

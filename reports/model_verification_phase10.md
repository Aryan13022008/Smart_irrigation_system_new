# Smart Agriculture — Phase 10: Deep Learning Models & Phase 9 Audit Report

**Project Directory:** `/Users/aryanagrawal/ESA_PROJECT`  
**Execution Date:** 2026-10-10  
**Phase:** Phase 10 — Independent Deep Learning Model Verification & Phase 9 Results Audit  

---

## Executive Summary

A comprehensive, read-only audit was conducted on the three deep learning sequence models (`improved_lstm.pt`, `tcn.pt`, `transformer.pt`) and the Phase 9 11-feature model training results.

1. **Deep Learning Checkpoints:** All three deep learning models load safely into memory, execute forward passes, and output valid probabilities $[0.0, 1.0]$.
2. **Provenance & Target Mismatch:** All three deep learning models were trained on [`data/irrigation_212223.csv`](file:///Users/aryanagrawal/ESA_PROJECT/data/irrigation_212223.csv) ($569$ rows) using a **synthetic standard-deviation target** (`synthetic_target = row_volatility > quantile(0.60)`) across 11 USDA research field station sensor plots (`C535CA`, `C55D7C`, etc.), **not real crop irrigation events or live ESP32 sensors**.
3. **Data Leakage in DL Pipelines:** All three DL training scripts (`train_lstm.py`, `train_tcn.py`, `train_transformer.py`) contain severe data leakage:
   - Global `StandardScaler.fit_transform()` was executed before splitting.
   - `np.random.shuffle()` was applied across overlapping 24-step sliding time sequences, allowing validation windows to peek at training windows.
   - TCN and Transformer evaluated **only on validation data with data leakage**; no independent test set was ever defined or evaluated.
4. **Phase 9 Audit:** The reported Phase 9 results were verified mathematically down to 6 decimal places from the exact test predictions. `.bfill()` only affected row index 0 of the training set ($0.0055\%$ of train data) and caused **zero leakage** into the test set.

---

## 1. Deep Learning Models Audit: LSTM, TCN, and Transformer

### 1.1 Improved LSTM (`training/models/improved_lstm.pt`)

- **Artifact Verification:**
  - File: [`training/models/improved_lstm.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/improved_lstm.pt) (Size: $225,713$ bytes).
  - Class: `ImprovedLSTMClassifier` (defined in [`training/inference.py#L24-L52`](file:///Users/aryanagrawal/ESA_PROJECT/training/inference.py#L24-L52) and [`training/train_lstm.py#L91-L119`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_lstm.py#L91-L119)).
  - Architecture: `LayerNorm(11) -> LSTM(input_size=11, hidden=64, layers=2, dropout=0.3) -> FC1(64->32) -> ReLU -> Dropout(0.3) -> FC2(32->1)`.
  - Input Shape: `(batch, 24, 11)`. Output Shape: `(batch, 1)`.
  - State dict loaded safely with 14 parameter tensors.
- **Training Provenance:**
  - Script: [`training/train_lstm.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_lstm.py).
  - Dataset: [`data/irrigation_212223.csv`](file:///Users/aryanagrawal/ESA_PROJECT/data/irrigation_212223.csv) ($569$ total rows).
  - Target: `synthetic_target` (binary indicator of whether spatial standard deviation across 11 field plot sensor IDs exceeds the 60th percentile).
  - Input Features: 11 USDA research station plot IDs (`C535CA`, `C55D7C`, `C54992`, `C535C9`, `C562C6`, `C54986`, `C58157`, `C54BB2`, `2F71D1`, `012384F3720B710DEE`, `C54762`).
- **Leakage Analysis:**
  - **Feature Scaling:** `StandardScaler` fitted on the entire dataset prior to splitting (`train_lstm.py#L59-L60`).
  - **Temporal Shuffling:** Sequences shuffled randomly with `np.random.shuffle(indices)` (`train_lstm.py#L75`), causing sliding-window temporal leakage.
- **Evaluation Verification:**
  - Evaluated post-hoc on the independent `iotsensordata.csv` test set ($N = 5,993$ samples).
  - **Test ROC-AUC:** **$0.5008$** (pure random chance / coin flip).
  - **Test PR-AUC:** **$0.5440$** (baseline positive prevalence: $54.38\%$).
  - **Tuned Threshold:** $0.3802$ (derived from validation set).
  - **Confusion Matrix at Tuned Threshold:**
    $$\begin{bmatrix} \text{TN: } 1 & \text{FP: } 2733 \\ \text{FN: } 1 & \text{TP: } 3258 \end{bmatrix}$$
  - Assessment: Fails completely on real irrigation data because its weights were optimized for spatial sensor volatility across research plots.
- **Inference Verification:**
  - Input `(1, 24, 11)` generates valid logits and finite probability $p \in [0.0, 1.0]$.
  - Missing sequence data cleanly returns `required: None` with explanation.
  - Invalid shape `(1, 10, 5)` caught with `RuntimeError: Given normalized_shape=[11], expected input with shape [*, 11], but got input of size [1, 10, 5]`.
- **Live Compatibility:**
  - **BLOCKED.** The ESP32 provides 2 capacitive probes and 1 DHT22. It cannot generate 11 distinct historical spatial plot channels.

---

### 1.2 TCN (`training/models/tcn.pt`)

- **Artifact Verification:**
  - File: [`training/models/tcn.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/tcn.pt) (Size: $796,045$ bytes).
  - Class: `TCNClassifier` (defined in [`training/inference.py#L82-L103`](file:///Users/aryanagrawal/ESA_PROJECT/training/inference.py#L82-L103) and [`training/train_tcn.py#L70-L91`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_tcn.py#L70-L91)).
  - Architecture: 3 Causal Dilated Residual Blocks with channels `(64, 64, 128)`, kernel size 5, dilations `[1, 2, 4]`, and linear head `128 -> 1`.
  - Input Shape: `(batch, 24, 11)` (internally transposed to `(batch, 11, 24)`). Output Shape: `(batch, 1)`.
  - State dict loaded safely with 18 parameter tensors.
- **Training Provenance:**
  - Script: [`training/train_tcn.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_tcn.py).
  - Dataset: [`data/irrigation_212223.csv`](file:///Users/aryanagrawal/ESA_PROJECT/data/irrigation_212223.csv) ($569$ rows).
  - Target: `synthetic_target` (spatial standard deviation across 11 field plot sensor IDs).
- **Leakage Analysis:**
  - Global `StandardScaler.fit_transform()` across all rows before split (`train_tcn.py#L141-L142`).
  - `np.random.shuffle(indices)` across overlapping sliding sequence windows (`train_tcn.py#L157`).
  - **No Held-Out Test Set:** Training evaluated only an 80/20 train/validation split.
- **Evaluation Verification:**
  - **Validation ROC-AUC:** $0.9107$ (recorded in [`training/models/tcn_metrics.json`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/tcn_metrics.json)).
  - **Validation PR-AUC:** $0.8603$.
  - **Optimal Threshold:** $0.9799$ (tuned directly on the validation set).
  - **Test Metrics:** **NONE.** No independent test set evaluation was ever performed. High validation scores are an artifact of random sequence shuffling.
- **Inference Verification:**
  - Input `(1, 24, 11)` executes cleanly and produces finite probability in $[0.0, 1.0]$.
  - Invalid shape `(1, 10, 5)` caught with `RuntimeError: Given groups=1, weight of size [64, 11, 1], expected input [1, 5, 10] to have 11 channels`.
- **Live Compatibility:**
  - **BLOCKED.** Incompatible with ESP32 single-station telemetry.

---

### 1.3 Transformer (`training/models/transformer.pt`)

- **Artifact Verification:**
  - File: [`training/models/transformer.pt`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/transformer.pt) (Size: $87,384$ bytes).
  - Class: `TransformerClassifier` (defined in [`training/inference.py#L125-L167`](file:///Users/aryanagrawal/ESA_PROJECT/training/inference.py#L125-L167) and [`training/train_transformer.py#L42-L84`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_transformer.py#L42-L84)).
  - Architecture: Linear projection `11 -> 32`, sinusoidal positional encoding, 2 Transformer Encoder layers (`d_model=32, nhead=4, dim_feedforward=64`), pooling head `32 -> 1`.
  - Input Shape: `(batch, 24, 11)`. Output Shape: `(batch, 1)`.
  - Checkpoint dict loaded safely (`state_dict`, `n_features: 11`, `d_model: 32`, `n_heads: 4`, `n_layers: 2`).
- **Training Provenance:**
  - Script: [`training/train_transformer.py`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_transformer.py).
  - Dataset: [`data/irrigation_212223.csv`](file:///Users/aryanagrawal/ESA_PROJECT/data/irrigation_212223.csv) ($569$ rows).
  - Target: `synthetic_target` (spatial standard deviation across 11 field plot sensor IDs).
- **Leakage Analysis:**
  - Global `StandardScaler.fit_transform()` across full dataset before split.
  - `np.random.shuffle(indices)` across overlapping sliding sequence windows.
  - No independent test set split.
- **Evaluation Verification:**
  - **Validation ROC-AUC:** $0.9538$ (recorded in [`training/models/transformer_metrics.json`](file:///Users/aryanagrawal/ESA_PROJECT/training/models/transformer_metrics.json)).
  - **Validation PR-AUC:** $0.9291$.
  - **Optimal Threshold:** $0.6001$.
  - **Test Metrics:** **NONE.** No test split evaluated.
- **Inference Verification:**
  - Input `(1, 24, 11)` executes cleanly and produces finite probability in $[0.0, 1.0]$.
  - Invalid shape `(1, 10, 5)` caught with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x5 and 11x32)`.
- **Live Compatibility:**
  - **BLOCKED.** Incompatible with ESP32 single-station telemetry.

---

### 1.4 Deep Learning Model Status Summary

| Model | Artifact Verification | Evaluation Verification | Inference Verification | Live Readiness | Operational Recommendation |
|:---|:---:|:---:|:---:|:---:|:---|
| **LSTM** | **VERIFIED** | **VERIFIED (0.5008 ROC-AUC)** | **VERIFIED** | 🔴 **BLOCKED** | Do not deploy (Random chance on irrigation) |
| **TCN** | **VERIFIED** | **UNVERIFIED (No Test Set)** | **VERIFIED** | 🔴 **BLOCKED** | Do not deploy (Synthetic spatial target) |
| **Transformer** | **VERIFIED** | **UNVERIFIED (No Test Set)** | **VERIFIED** | 🔴 **BLOCKED** | Do not deploy (Synthetic spatial target) |

---

## 2. Audit of Phase 9 Results (`random_forest_11feat`)

### 2.1 `.bfill()` and Boundary Leakage Audit
Inspected [`training/train_11feat_model.py#L90-L105`](file:///Users/aryanagrawal/ESA_PROJECT/training/train_11feat_model.py#L90-L105):

- **Lag Construction:**
  `df["soil_moisture_previous"] = df["soil_moisture"].shift(1)`
- **NaN Profile:**
  - Across all $23,995$ cleaned rows, exactly **1 row** contained `NaN`: row index `0` (the first timestamp, 2022-01-01 00:00:00).
- **`.bfill()` Impact:**
  - `.bfill()` imputed row index `0` using the values from row index `1` (1 hour in the future).
  - Row index `0` resides entirely within the **training split** ($17,996$ rows). It accounts for $0.0055\%$ of the training data.
  - **Zero Test Leakage:** Row index `0` has no interaction with the test partition ($5,999$ rows, indices $17,996$ to $23,994$).
- **Split Boundary Continuity:**
  - At the split boundary (test row index $17,996$), its 1-hour lag value correctly accesses training row index $17,995$. This represents authentic historical backward continuity.

---

### 2.2 Mathematical Recalculation of Phase 9 Metrics

Recalculated directly from confusion matrices and raw probabilities on the held-out test set ($N = 5,999$ samples):

#### At Default Threshold (0.50):
- **Confusion Matrix:** $\begin{bmatrix} \text{TN: } 1385 & \text{FP: } 1354 \\ \text{FN: } 642 & \text{TP: } 2618 \end{bmatrix}$ (Total: $5,999$)
- **Accuracy:** $\frac{2618 + 1385}{5999} = \mathbf{0.667278}$ (Reported: $0.667278$) $\to$ **EXACT MATCH**
- **Precision:** $\frac{2618}{2618 + 1354} = \mathbf{0.659114}$ (Reported: $0.659114$) $\to$ **EXACT MATCH**
- **Recall:** $\frac{2618}{2618 + 642} = \mathbf{0.803067}$ (Reported: $0.803067$) $\to$ **EXACT MATCH**
- **F1-Score:** $\frac{2 \times 0.659114 \times 0.803067}{0.659114 + 0.803067} = \mathbf{0.724004}$ (Reported: $0.724004$) $\to$ **EXACT MATCH**
- **ROC-AUC:** $\mathbf{0.680756}$ $\to$ **EXACT MATCH**
- **PR-AUC:** $\mathbf{0.681681}$ $\to$ **EXACT MATCH**

#### At Validation-Tuned Threshold (0.3271):
- **Confusion Matrix:** $\begin{bmatrix} \text{TN: } 1068 & \text{FP: } 1671 \\ \text{FN: } 458 & \text{TP: } 2802 \end{bmatrix}$ (Total: $5,999$)
- **Accuracy:** $\frac{2802 + 1068}{5999} = \mathbf{0.645108}$ (Reported: $0.645108$) $\to$ **EXACT MATCH**
- **Precision:** $\frac{2802}{2802 + 1671} = \mathbf{0.626425}$ (Reported: $0.626425$) $\to$ **EXACT MATCH**
- **Recall:** $\frac{2802}{2802 + 458} = \mathbf{0.859509}$ (Reported: $0.859509$) $\to$ **EXACT MATCH**
- **F1-Score:** $\frac{2 \times 0.626425 \times 0.859509}{0.626425 + 0.859509} = \mathbf{0.724686}$ (Reported: $0.724686$) $\to$ **EXACT MATCH**
- **ROC-AUC:** $\mathbf{0.680756}$ $\to$ **EXACT MATCH**
- **PR-AUC:** $\mathbf{0.681681}$ $\to$ **EXACT MATCH**

---

### 2.3 Definition & Threshold Consistency Check

- **Test Split Consistency:** Both the 13-feature baseline and the 11-feature model were evaluated on the identical $5,999$ chronologically held-out test samples.
- **Threshold Tuning Consistency:** Both models tuned their decision threshold strictly on an internal chronological validation split carved from the training split ($3,599$ rows) using F1-maximization without peeking at test labels.
- **Metric Definitions:** Both used identical scikit-learn implementations for ROC-AUC, PR-AUC, accuracy, precision, recall, and F1-score.

---

## 3. Test Suite Verification

Executed full regression suite:
```bash
python3 -m unittest test_ml_integration.py
```
**Output:**
```
Ran 25 tests in 2.676s

OK
```
All 25 unit and integration tests passed, confirming that the model audit and Phase 9 additions did not alter existing system behaviors or safety checks.

---

## 4. Unresolved Blockers and Recommended Next Steps

### Unresolved Blockers:
1. **Deep Learning Incompatibility:** The LSTM, TCN, and Transformer models are fundamentally unusable for this project. They were trained on 11 research station field plot channels with a synthetic target.
2. **Reduced Model Degradation:** The 11-feature model drops ROC-AUC to $0.6808$ (from $0.7927$) due to missing root-zone soil temperature.
3. **Physical Soil Moisture Equivalence:** The ESP32's capacitive probes provide a two-point relative index ($0\% =$ air, $100\% =$ water), whereas the training data decision boundary expects physical soil moisture where $\le 59.5\%$ triggers irrigation.

### Single Recommended Next Step:
**Permanently retire/deprecate the LSTM, TCN, and Transformer models from the active irrigation pipeline, and install an inexpensive DS18B20 soil temperature probe on the ESP32 to restore the 13-feature baseline model to full readiness.**

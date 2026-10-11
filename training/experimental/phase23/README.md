# Phase 23 — Live Sensor Data Readiness & Forecasting Bridge Audit

## Overview
This directory contains isolated telemetry sequence-buffer validation routines, data readiness diagnostics, and bridge safety checks for the Smart Agriculture System (`ESA_PROJECT`).

**Objective:** Bridge real hardware readings to deep learning models (LSTM, TCN, Transformer) safely without modifying production models, breaking safety locks, or fabricating missing data.

---

## Safety and Scope Rules
- `PUMP_ALLOWED = False` is strictly hard-coded. Zero actuator or relay functionality exists.
- The Phase 11 production 11-feature model (`training/models/random_forest_11feat.pkl`, threshold `0.3270985223048995`) remains the sole active decision model for production irrigation advisory.
- Missing sensor telemetry is **never fabricated, simulated, or imputed** with synthetic constants.
- High-frequency serial packets (2.5-second loop) are **never treated as independent 10-minute sequence observations**.
- No models are trained or retrained.

---

## Module Index
- `sequence_buffer.py`:
  - `validate_and_construct_sequence()`: Inspects candidate historical observations, verifies monotonic ordering, validates 10-minute cadence ($600 \pm 60$ s), checks probe identity consistency, ensures complete 14-feature availability, and constructs a clean 24-step sequence tensor for model input. Returns descriptive blocker codes (`HIGH_FREQUENCY_PACKET_STREAM`, `INSUFFICIENT_OBSERVATIONS`, `IRREGULAR_TEMPORAL_SPACING`, `MIXED_PROBE_IDENTITIES`, `MISSING_FEATURE_COLUMNS`) if any rule is violated.
  - `audit_live_telemetry_history()`: Directly queries `data/sensor_history.db` and audits whether existing SQLite records satisfy sequence requirements.

---

## Telemetry Audit Findings
1. **Sampling Frequency vs Model Cadence:**
   The ESP32 firmware emits packets every 2.5 seconds. SQLite logs packets as received by the host loop. While over 6,500 records are stored across ~21.7 hours, they are clustered in high-frequency bursts (median $\Delta t \approx 2.52$s) rather than a scheduled 10-minute cadence.
2. **Missing Channels:**
   The ESP32 station only provides relative probe ADC (Probe 1 on GPIO 32, Probe 2 on GPIO 33) and DHT22 air temp/humidity. It lacks 8 of the 14 mandatory channels required by the sequence models (Atmospheric Pressure, Wind Speed, Solar Radiation, Dual-depth Soil Temperatures, Electrical Conductivity, and Soil pH).
3. **Soil Moisture Calibration Decoupling:**
   Current probes use an air/water-relative calibration index:
   $$\text{Moisture \%} = \text{clamp}\left(0, 100, \frac{\text{dry\_adc} - \text{raw\_adc}}{\text{dry\_adc} - \text{wet\_adc}} \times 100\right)$$
   This relative index is fundamentally distinct from the training dataset's volumetric water content (VWC).

---

## Current Operational Status
**`LIVE FORECAST UNAVAILABLE — INPUT DATA NOT COMPATIBLE WITH TRAINED MODEL`**
Inference on live ESP32 data is strictly blocked by `validate_and_construct_sequence()`. Only verified offline reference sequences from the research dataset are permitted for demonstration.

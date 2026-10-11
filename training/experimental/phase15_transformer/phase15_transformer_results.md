# Phase 15 Experimental Transformer Soil-Moisture Forecasting

**Scope:** Isolated experimental research only. The Transformer does NOT predict irrigation ON/OFF decisions and is NOT integrated into production. Pump control remains strictly disabled.

- Target: `target_mean_24h` (24-hour mean soil moisture)
- Input: 24 contiguous 10-minute observations (4 hours)
- Features: weather_temp, weather_humidity, weather_rain, weather_pressure, weather_wind_speed, weather_radiation, soil_temperature_0-7cm, soil_temperature_7-18cm, ec, ph, soil_moisture, irrigation_duration_minutes, hour, month
- Windows: Train=14208, Val=6326, Test=6348
- Scaling: StandardScaler fitted strictly on training rows
- Splitting: Chronological within each sector (70/15/15); no gap or boundary crossing

## Overall Results (Held-Out Test Set, N = 6,348)

| Model | Parameters | Test MAE (%) | Test RMSE (%) | Pooled Test R² |
|:---|:---:|:---:|:---:|:---:|
| **Persistence Baseline (Current Moisture)** | **0** | **0.4124** | **0.7641** | **0.99863** |
| **Experimental Transformer** | **18,113** | **6.1237** | **8.7057** | **0.82219** |
| Experimental LSTM | 6,689 | 3.8665 | 5.1434 | 0.93794 |
| Experimental TCN | 17,921 | 4.6904 | 7.6977 | 0.86098 |

## Per-Zone Test Metrics Breakdown

| Sector | Samples | Target Mean ± Std | Persistence MAE | Transformer MAE | LSTM MAE | TCN MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sector 1 (Tomato Field)** | 1408 | 19.26 ± 0.16 | **0.4678** | 0.8398 | 1.4221 | 2.2372 |
| **Sector 2 (Tomato Field)** | 1721 | 15.53 ± 0.26 | **0.3810** | 1.9944 | 2.6545 | 2.1904 |
| **Sector 4 (Zucchini)** | 1599 | 57.27 ± 1.89 | **0.3807** | 5.4820 | 6.2999 | 1.6669 |
| **Sector 5 (Blueberry)** | 1620 | 2.20 ± 1.97 | **0.4290** | 15.7363 | 4.8766 | 12.4626 |

## Key Findings & Conclusion

1. **Persistence Baseline Remains Dominant:** The persistence baseline achieves MAE 0.4124, outperforming the Transformer by 8.5x (Transformer MAE: 6.1237).
2. **Post-Season Test Equilibrium:** In the late-season test period, zero irrigation occurred across all sectors. Soil moisture was in static equilibrium, rendering persistence optimal.
3. **Simpson's Paradox in Pooled R²:** Pooled R² appears high (0.8222) due to between-sector moisture variance (~2% to 57%), but within individual sectors, neural network R² is negative.
4. **Production Verdict:** The Transformer does NOT beat the baseline and must NOT be deployed to production.

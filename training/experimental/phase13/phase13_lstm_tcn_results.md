# Phase 13 Experimental LSTM/TCN Training

**Scope:** experimental soil-moisture forecasting only. These models do not predict irrigation-needed ON/OFF. They are not integrated into production. Dataset provenance and target alignment remain unresolved.

- Target: `target_mean_24h`
- Input: 24 contiguous 10-minute observations (240 minutes)
- Features: weather_temp, weather_humidity, weather_rain, weather_pressure, weather_wind_speed, weather_radiation, soil_temperature_0-7cm, soil_temperature_7-18cm, ec, ph, soil_moisture, irrigation_duration_minutes, hour, month
- Window counts: {'train': 14208, 'val': 6327, 'test': 6347}
- Scaling: fit on training rows only
- Split: chronological within each sector; windows cannot cross split boundaries or gaps

## LSTM

- Parameters: 6689
- Epochs: 13 (early stopping)
- Best validation MSE: 20.7685
- Test MAE: 3.8670
- Test RMSE: 5.1438
- Test R²: 0.9379
- Persistence baseline (current soil moisture) MAE: 0.4124

## TCN

- Parameters: 17921
- Epochs: 23 (early stopping)
- Best validation MSE: 27.0696
- Test MAE: 4.6909
- Test RMSE: 7.6983
- Test R²: 0.8610
- Persistence baseline (current soil moisture) MAE: 0.4124

## Limitations and safety

- Metrics apply only to the supplied processed dataset and its `target_mean_24h`; target construction and independent dataset provenance are not established.
- This is not an irrigation-needed classifier and cannot establish field performance in Tamil Nadu.
- Model checkpoints are experimental artifacts only. Phase 11 model, dashboard, calibration, database, firmware, and pump lock were not modified.

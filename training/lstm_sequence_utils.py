import numpy as np
import pandas as pd

def create_lstm_sequences(df, feature_cols, target_col='target', window_size=24, forecast_horizon=6):
    """
    Creates 3D sequences [Samples, Window Size, Features] and a future-shifted target.
    """
    data = df[feature_cols].values
    target = df[target_col].values
    
    X, y = [], []
    
    # Iterate with enough buffer for the window and future forecasting horizon
    for i in range(window_size, len(df) - forecast_horizon):
        # Past window of features (e.g., last 24 time steps)
        X.append(data[i - window_size : i])
        
        # Future target horizon: looking ahead by 'forecast_horizon' steps
        future_window = target[i : i + forecast_horizon]
        y.append(1 if np.any(future_window == 1) else 0)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
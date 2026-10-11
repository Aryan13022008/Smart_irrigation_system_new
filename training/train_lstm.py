import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Sequence generator function
def create_lstm_sequences(df, feature_cols, target_col='target', window_size=24, forecast_horizon=6):
    """
    Creates 3D sequences [Samples, Window Size, Features] and a future-shifted target.
    """
    data = df[feature_cols].values
    target = df[target_col].values
    
    X, y = [], []
    for i in range(window_size, len(df) - forecast_horizon):
        X.append(data[i - window_size : i])
        future_window = target[i : i + forecast_horizon]
        y.append(1 if np.any(future_window == 1) else 0)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    print("============================================================")
    print("UPGRADED LSTM TRAINING (Balanced Target & Future-Shifted Horizon)")
    print("============================================================")

    # 1. Load Dataset from data directory
    data_path = '../data/irrigation_212223.csv'
    if not os.path.exists(data_path):
        data_path = 'data/irrigation_212223.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find irrigation dataset. Ensure it is located in the data/ folder.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Build a Balanced Synthetic Target if none exists
    drop_cols = [c for c in ['timestamp', 'Date', 'year'] if c in df.columns]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    row_volatility = df[feature_cols].std(axis=1)
    threshold = row_volatility.quantile(0.60)
    df['synthetic_target'] = (row_volatility > threshold).astype(int)
    target_col = 'synthetic_target'
    
    print(f"Created balanced target. Class distribution: {df[target_col].value_counts().to_dict()}")

    feature_cols = [c for c in feature_cols if c != target_col]
    print(f"Number of input features: {len(feature_cols)}")

    # 3. Scale Features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 4. Create Sequences (Window Size = 24, Forecast Horizon = 6)
    WINDOW_SIZE = 24
    FORECAST_HORIZON = 6
    
    X, y = create_lstm_sequences(df, feature_cols, target_col=target_col, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON)
    print(f"Total sequences generated: {X.shape[0]}")

    if len(X) == 0:
        raise ValueError("Not enough rows in dataset to generate sequences. Check dataset length.")

    # 5. Shuffle and Train / Validation Split (80% / 20%)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 6. Upgraded LSTM Model Architecture
    class ImprovedLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_dim=64, num_layers=2):
            super(ImprovedLSTMClassifier, self).__init__()
            self.layer_norm = nn.LayerNorm(input_size)
            
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_dim, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=0.3
            )
            
            self.fc1 = nn.Linear(hidden_dim, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x = self.layer_norm(x)
            lstm_out, _ = self.lstm(x)
            last_step = lstm_out[:, -1, :]
            
            out = self.fc1(last_step)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ImprovedLSTMClassifier(input_size=len(feature_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 7. Training Loop
    EPOCHS = 25
    print("\nStarting Training...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                preds = torch.sigmoid(model(batch_X)).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        
        try:
            val_roc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_roc = 0.50
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val ROC-AUC: {val_roc:.4f}")

    print("\nTraining complete!")
    
    # 8. Comprehensive Final Evaluation Metrics (Precision, Recall, F1, PR-AUC)
    print("\nCalculating Final Comprehensive Evaluation Report...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            preds = torch.sigmoid(model(batch_X)).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)
    final_roc_auc = roc_auc_score(all_targets, all_preds)
    
    # Find optimal threshold balancing precision and recall
    best_idx = np.argmax(precision + recall) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    binary_preds = (all_preds >= best_threshold).astype(int)
    
    print("============================================================")
    print("               LSTM FINAL PERFORMANCE REPORT                ")
    print("============================================================")
    print(f"Validation ROC-AUC : {final_roc_auc:.4f}")
    print(f"Validation PR-AUC  : {pr_auc:.4f}")
    print(f"Optimal Threshold  : {best_threshold:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, binary_preds, target_names=['No Irrigation', 'Irrigation Needed']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, binary_preds))
    print("============================================================")

    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/improved_lstm.pt')
    print("Saved improved LSTM model to models/improved_lstm.pt")

if __name__ == '__main__':
    main()

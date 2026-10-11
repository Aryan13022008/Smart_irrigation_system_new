"""
Shared PyTorch training/evaluation loop used by train_lstm.py,
train_tcn.py, and train_transformer.py so the three models are
trained and scored in exactly the same way, and are directly
comparable to each other and to the Random Forest baseline.
"""

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch_size=32, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, y_train, epochs=80, lr=1e-3, weight_decay=1e-4, val_loader=None, patience=15):
    """
    Trains with BCEWithLogitsLoss using pos_weight. 
    Now includes PR-AUC evaluated early stopping to prevent severe overfitting.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = None
    if val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    model.to(DEVICE)
    
    best_pr_auc = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        
        if val_loader is not None:
            model.eval()
            val_logits, val_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    val_logits.append(model(xb).squeeze(-1).cpu())
                    val_labels.append(yb)
            
            y_val_true = torch.cat(val_labels).numpy()
            y_val_proba = expit(torch.cat(val_logits).numpy())
            val_pr_auc = average_precision_score(y_val_true, y_val_proba)
            
            scheduler.step(val_pr_auc)
            
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs} - train loss: {avg_loss:.4f} - val PR-AUC: {val_pr_auc:.4f}")
                
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best Val PR-AUC: {best_pr_auc:.4f}")
                break
        else:
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs} - train loss: {avg_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model


@torch.no_grad()
def predict_proba(model, loader):
    """Returns (y_true, y_proba) as numpy arrays — the raw sigmoid
    probabilities, with no threshold applied yet. Used both by
    evaluate_model (at threshold=0.5 or a chosen threshold) and by
    threshold-tuning code that needs the full probability spread."""
    model.eval()
    all_logits, all_labels = [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb).squeeze(-1).cpu()
        all_logits.append(logits)
        all_labels.append(yb)

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    # expit = numerically stable sigmoid (plain 1/(1+exp(-x)) overflows
    # for very negative logits, which extreme pos_weight can produce).
    y_proba = expit(logits)
    return y_true, y_proba


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(title, m):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}  <-- most important for this project")
    print(f"F1-score : {m['f1']:.4f}")
    print(f"ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"PR-AUC   : {m['pr_auc']:.4f}")
    print("\nConfusion Matrix ([[TN, FP], [FN, TP]]):")
    print(np.array(m["confusion_matrix"]))


def evaluate_model(model, test_loader, threshold=0.5, title="TEST SET EVALUATION"):
    """Kept for backwards compatibility with the original single-
    threshold flow. New scripts should prefer predict_proba() +
    compute_metrics() so they can also evaluate at a tuned threshold."""
    y_true, y_proba = predict_proba(model, test_loader)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(title, metrics)
    return metrics

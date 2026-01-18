import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 0) Reproducibility helpers
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may slow a bit but improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1) Models 
# =========================
class Attention(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x):
        # x: [batch, feature_dim]
        scores = x @ self.attention_weights      # [batch]
        scores = scores.unsqueeze(-1)            # [batch, 1]
        scores = torch.softmax(scores, dim=1)    # still [batch, 1] but stable
        scores = scores.squeeze(-1)              # [batch]
        return x * scores.unsqueeze(-1)          # [batch, feature_dim]


class DNNWithAttention(nn.Module):
    def __init__(self, input_dim: int, n_units: int = 256, dropout_p: float = 0.5):
        super().__init__()
        self.attention = Attention(input_dim)
        self.fc1 = nn.Linear(input_dim, n_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(n_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.attention(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class FusionNN(nn.Module):
    def __init__(self, hidden: int = 64, dropout_p: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)  # z=[P1,P2]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.sigmoid(self.fc2(z))
        return z


# =========================
# 2) Training utilities
# =========================
def train_dnn(model, train_loader, val_loader, device,
              lr=0.01, weight_decay=1e-4, max_epochs=300, patience=20):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(max_epochs):
        # train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return model


def train_fusion(fusion_model, z_train, y_train, device, lr=1e-3, epochs=200):
    criterion = nn.BCELoss()
    opt = optim.Adam(fusion_model.parameters(), lr=lr)
    fusion_model.train()

    z_train = z_train.to(device)
    y_train = y_train.to(device)

    for _ in range(epochs):
        opt.zero_grad()
        out = fusion_model(z_train)
        loss = criterion(out, y_train)
        loss.backward()
        opt.step()

    return fusion_model


# =========================
# 3) One repeated 80/20 run
# =========================
def run_one_holdout_repeat(df, features_ch1, features_ch2, target,
                           split_seed: int, device):
    """
    One stratified 80/20 evaluation (Fusion model).
    Validation = 20% of training only (for DNN early stopping).
    """
    set_seed(split_seed)

    X1 = df[features_ch1].values
    X2 = df[features_ch2].values
    y  = df[target].values

    # One consistent split index for BOTH channels
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X1, X2, y, test_size=0.2, random_state=split_seed, stratify=y
    )

    # DNN: split training into train/val (val from training only)
    X2_tr2, X2_val, y_tr2, y_val = train_test_split(
        X2_tr, y_tr, test_size=0.2, random_state=split_seed, stratify=y_tr
    )

    # For clean fusion training, we use the SAME rows for Channel-1 and Channel-2 training subset:
    # We align Channel-1 training subset with the DNN train subset size by splitting X1_tr the same way:
    X1_tr2, X1_val_dummy, y1_tr2, _ = train_test_split(
        X1_tr, y_tr, test_size=0.2, random_state=split_seed, stratify=y_tr
    )
    # Note: we do not need X1_val_dummy; it's only to match indices behavior.

    # Standardization (fit on training only)
    scaler1 = StandardScaler().fit(X1_tr2)
    X1_tr2_s = scaler1.transform(X1_tr2)
    X1_te_s  = scaler1.transform(X1_te)

    scaler2 = StandardScaler().fit(X2_tr2)
    X2_tr2_s = scaler2.transform(X2_tr2)
    X2_val_s = scaler2.transform(X2_val)
    X2_te_s  = scaler2.transform(X2_te)

    # ---- Channel 1: XGB + SVM (soft voting) ----
    model_xgb = xgb.XGBClassifier(
        learning_rate=0.25, max_depth=3, n_estimators=500, subsample=0.8,
        eval_metric='logloss', random_state=split_seed
    )
    model_svm = SVC(C=100, gamma=100, kernel='rbf', probability=True)

    ensemble = VotingClassifier(
        estimators=[('xgb', model_xgb), ('svm', model_svm)],
        voting='soft'
    )
    ensemble.fit(X1_tr2_s, y1_tr2)

    P1_tr = ensemble.predict_proba(X1_tr2_s)[:, 1]
    P1_te = ensemble.predict_proba(X1_te_s)[:, 1]

    # ---- Channel 2: DNN + attention ----
    X2_tr_tensor  = torch.tensor(X2_tr2_s, dtype=torch.float32)
    y2_tr_tensor  = torch.tensor(y_tr2, dtype=torch.float32).view(-1, 1)
    X2_val_tensor = torch.tensor(X2_val_s, dtype=torch.float32)
    y2_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X2_te_tensor  = torch.tensor(X2_te_s, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X2_tr_tensor, y2_tr_tensor), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X2_val_tensor, y2_val_tensor), batch_size=64, shuffle=False)

    dnn = DNNWithAttention(input_dim=X2_tr2_s.shape[1], n_units=256, dropout_p=0.5).to(device)
    dnn = train_dnn(dnn, train_loader, val_loader, device, lr=0.01, max_epochs=300, patience=20)

    dnn.eval()
    with torch.no_grad():
        P2_tr = dnn(X2_tr_tensor.to(device)).cpu().numpy().squeeze()
        P2_te = dnn(X2_te_tensor.to(device)).cpu().numpy().squeeze()

    # ---- Fusion model ----
    Z_tr = torch.tensor(np.column_stack([P1_tr, P2_tr]), dtype=torch.float32)
    y_f_tr = torch.tensor(y1_tr2, dtype=torch.float32).view(-1, 1)

    Z_te = torch.tensor(np.column_stack([P1_te, P2_te]), dtype=torch.float32)
    y_te_np = y_te.astype(int)

    fusion = FusionNN(hidden=64, dropout_p=0.5).to(device)
    fusion = train_fusion(fusion, Z_tr, y_f_tr, device, lr=1e-3, epochs=200)

    fusion.eval()
    with torch.no_grad():
        prob_te = fusion(Z_te.to(device)).cpu().numpy().squeeze()
    pred_te = (prob_te >= 0.5).astype(int)

    # Metrics
    out = {
        "ACC": accuracy_score(y_te_np, pred_te),
        "P": precision_score(y_te_np, pred_te, zero_division=0),
        "R": recall_score(y_te_np, pred_te, zero_division=0),
        "F1": f1_score(y_te_np, pred_te, zero_division=0),
        "AUC": roc_auc_score(y_te_np, prob_te),
    }
    return out


# =========================
# 4) Repeated 80/20 experiment
# =========================
def summarize_metrics(df_metrics: pd.DataFrame):
    # mean ± std
    mean = df_metrics.mean()
    std  = df_metrics.std(ddof=1)

    # 95% CI for the mean using normal approx (good for n>=30)
    n = len(df_metrics)
    ci_half = 1.96 * (std / np.sqrt(n))

    summary = pd.DataFrame({
        "mean": mean,
        "std": std,
        "95%_CI_low": mean - ci_half,
        "95%_CI_high": mean + ci_half
    })
    return summary


if __name__ == "__main__":
    # ====== USER SETTINGS ======
    CSV_PATH = 'data/HeartCT.csv'  # dataset file
    N_REPEATS = 20                 # Step (1): 30 repeated hold-out splits
    BASE_SEED = 32                 # starting seed
    # ===========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(CSV_PATH)

    # Channels (same as my paper)
    features_channel_1 = ['age', 'sex', 'chol', 'trestbps', 'fbs']
    features_channel_2 = ['cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    target = 'target'

    all_runs = []
    for i in range(N_REPEATS):
        seed_i = BASE_SEED + i
        metrics_i = run_one_holdout_repeat(
            df, features_channel_1, features_channel_2, target,
            split_seed=seed_i, device=device
        )
        metrics_i["seed"] = seed_i
        all_runs.append(metrics_i)
        print(f"[Repeat {i+1}/{N_REPEATS}] seed={seed_i} | "
              f"ACC={metrics_i['ACC']:.4f}, P={metrics_i['P']:.4f}, R={metrics_i['R']:.4f}, F1={metrics_i['F1']:.4f}, AUC={metrics_i['AUC']:.4f}")

    df_metrics = pd.DataFrame(all_runs).set_index("seed")
    df_metrics.to_csv("holdout_repeats_metrics.csv")

    summary = summarize_metrics(df_metrics)
    print("\n==== Repeated 80/20 Hold-out Summary (mean ± std, 95% CI for mean) ====")
    print(summary)

    # print in percentages for paper convenience
    percent = summary.copy()
    for c in ["mean", "std", "95%_CI_low", "95%_CI_high"]:
        percent[c] = 100.0 * percent[c]
    print("\n==== Same summary in % ====")
    print(percent)

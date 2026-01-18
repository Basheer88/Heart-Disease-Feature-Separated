# Dual-Channel Heart Disease Diagnosis (Single Split OR 10-Fold CV with Leakage-Safe Fusion)
# - Channel 1: (XGBoost + SVM) soft-voting ensemble
# - Channel 2: DNN with simple attention
# - Fusion: FusionNN trained on OUT-OF-FOLD (OOF) base-model predictions (Option 2, leakage-safe)

import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
RUN_CV = False          # True => 10-fold CV (leakage-safe fusion); False => single split pipeline
N_SPLITS = 10           # 10-fold 
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DNN training knobs
DNN_MAX_EPOCHS = 300
DNN_BATCH_SIZE = 32
DNN_LR = 0.01
DNN_WEIGHT_DECAY = 1e-4
DNN_HIDDEN_UNITS = 256

# Fusion training knobs
FUSION_EPOCHS = 300
FUSION_LR = 1e-3


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss: float):
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0

        if self.early_stop and self.verbose:
            print("Early stopping triggered")


# =========================
# MODELS
# =========================
class Attention(nn.Module):
    """
    Lightweight attention that outputs per-sample weights over features.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        scores = x @ self.attention_weights  # [B]
        scores = scores.unsqueeze(-1)        # [B, 1]
        scores = torch.softmax(scores, dim=1)  # softmax over dim=1 (single column -> yields 1s)
        scores = scores.squeeze(-1)            # [B]
        return x * scores.unsqueeze(-1)        # [B, D]


class DNNWithAttention(nn.Module):
    def __init__(self, input_dim: int, n_units: int):
        super().__init__()
        self.attention = Attention(input_dim)
        self.fc1 = nn.Linear(input_dim, n_units)
        self.fc2 = nn.Linear(n_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class FusionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# =========================
# TRAIN 
# =========================
def train_channel1_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
    model_xgb = xgb.XGBClassifier(
        learning_rate=0.25, max_depth=3, n_estimators=500, subsample=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE
    )
    model_svm = SVC(C=100, gamma=100, kernel="rbf", probability=True)
    ensemble = VotingClassifier(
        estimators=[("xgb", model_xgb), ("svm", model_svm)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def channel1_predict_proba(model: VotingClassifier, X: np.ndarray) -> np.ndarray:
    # positive class probability
    return model.predict_proba(X)[:, 1]


def train_channel2_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> DNNWithAttention:
    model = DNNWithAttention(input_dim=X_train.shape[1], n_units=DNN_HIDDEN_UNITS).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=DNN_LR, weight_decay=DNN_WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    early_stopping = EarlyStopping(patience=20, verbose=False)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )

    train_loader = DataLoader(train_ds, batch_size=DNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DNN_BATCH_SIZE, shuffle=False)

    for epoch in range(DNN_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item()

        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return model


@torch.no_grad()
def channel2_predict_proba(model: DNNWithAttention, X: np.ndarray) -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    probs = model(xb).squeeze().detach().cpu().numpy()
    return probs


def train_fusion_model(X_fused: np.ndarray, y: np.ndarray) -> FusionNN:
    fusion = FusionNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(fusion.parameters(), lr=FUSION_LR)

    X_t = torch.tensor(X_fused, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)

    fusion.train()
    for _ in range(FUSION_EPOCHS):
        optimizer.zero_grad()
        out = fusion(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    return fusion


@torch.no_grad()
def fusion_predict_proba(fusion: FusionNN, X_fused: np.ndarray) -> np.ndarray:
    fusion.eval()
    xb = torch.tensor(X_fused, dtype=torch.float32).to(DEVICE)
    probs = fusion(xb).squeeze().detach().cpu().numpy()
    return probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC only if both classes present
    if len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["auc"] = float("nan")
    return out


# =========================
# PIPELINE A: SINGLE SPLIT
# =========================
def run_single_split(df: pd.DataFrame) -> dict:
    features_channel_1 = ["age", "sex", "chol", "trestbps", "fbs"]
    features_channel_2 = ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    target = "target"

    X1 = df[features_channel_1].values
    X2 = df[features_channel_2].values
    y = df[target].values.astype(int)

    # One consistent split for both channels
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    X1_train, X1_test = X1[idx_train], X1[idx_test]
    X2_train, X2_test = X2[idx_train], X2[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Channel 1 scaling + model
    scaler1 = StandardScaler().fit(X1_train)
    X1_train_s = scaler1.transform(X1_train)
    X1_test_s = scaler1.transform(X1_test)

    ch1_model = train_channel1_ensemble(X1_train_s, y_train)
    ch1_train_prob = channel1_predict_proba(ch1_model, X1_train_s)
    ch1_test_prob = channel1_predict_proba(ch1_model, X1_test_s)

    # Channel 2 split train->train/val (for early stopping)
    idx_tr2, idx_val2 = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    X2_tr, X2_val = X2_train[idx_tr2], X2_train[idx_val2]
    y2_tr, y2_val = y_train[idx_tr2], y_train[idx_val2]

    scaler2 = StandardScaler().fit(X2_tr)
    X2_tr_s = scaler2.transform(X2_tr)
    X2_val_s = scaler2.transform(X2_val)
    X2_train_s = scaler2.transform(X2_train)
    X2_test_s = scaler2.transform(X2_test)

    ch2_model = train_channel2_dnn(X2_tr_s, y2_tr, X2_val_s, y2_val)
    ch2_train_prob = channel2_predict_proba(ch2_model, X2_train_s)
    ch2_test_prob = channel2_predict_proba(ch2_model, X2_test_s)

    # Fusion training on training set predictions (single-split setting)
    X_train_fused = np.column_stack([ch1_train_prob, ch2_train_prob])
    X_test_fused = np.column_stack([ch1_test_prob, ch2_test_prob])

    fusion = train_fusion_model(X_train_fused, y_train)
    y_test_prob = fusion_predict_proba(fusion, X_test_fused)

    metrics = compute_metrics(y_test, y_test_prob)

    # Optional: ROC curve plot (single split)
    try:
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        auc = roc_auc_score(y_test, y_test_prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2, label=f"Fusion (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Fusion Model)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    return metrics


# =========================
# PIPELINE B: 10-FOLD CV (leakage-safe fusion)
# =========================
def run_leakage_safe_cv(df: pd.DataFrame, n_splits: int = 10) -> dict:
    """
    Outer CV evaluates final fusion generalization.
    Inner CV (on outer-train) generates OOF predictions for base models to train FusionNN without leakage.
    Then, base models are refit on full outer-train to produce outer-test probabilities for fusion evaluation.
    """
    features_channel_1 = ["age", "sex", "chol", "trestbps", "fbs"]
    features_channel_2 = ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    target = "target"

    X1_all = df[features_channel_1].values
    X2_all = df[features_channel_2].values
    y_all = df[target].values.astype(int)

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(outer.split(X1_all, y_all), 1):
        print(f"\n===== OUTER FOLD {outer_fold}/{n_splits} =====")

        X1_tr, X1_te = X1_all[outer_tr_idx], X1_all[outer_te_idx]
        X2_tr, X2_te = X2_all[outer_tr_idx], X2_all[outer_te_idx]
        y_tr, y_te = y_all[outer_tr_idx], y_all[outer_te_idx]

        # ---- INNER CV: build OOF preds on outer-train for leakage-safe fusion training
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        oof_ch1 = np.zeros(len(y_tr), dtype=float)
        oof_ch2 = np.zeros(len(y_tr), dtype=float)

        for inner_fold, (in_tr, in_val) in enumerate(inner.split(X1_tr, y_tr), 1):
            # Split inner train/val
            X1_in_tr, X1_in_val = X1_tr[in_tr], X1_tr[in_val]
            X2_in_tr, X2_in_val = X2_tr[in_tr], X2_tr[in_val]
            y_in_tr, y_in_val = y_tr[in_tr], y_tr[in_val]

            # Channel 1 scaler fit on inner-train only
            sc1 = StandardScaler().fit(X1_in_tr)
            X1_in_tr_s = sc1.transform(X1_in_tr)
            X1_in_val_s = sc1.transform(X1_in_val)

            ch1 = train_channel1_ensemble(X1_in_tr_s, y_in_tr)
            oof_ch1[in_val] = channel1_predict_proba(ch1, X1_in_val_s)

            # Channel 2 scaler fit on inner-train only
            sc2 = StandardScaler().fit(X2_in_tr)
            X2_in_tr_s = sc2.transform(X2_in_tr)
            X2_in_val_s = sc2.transform(X2_in_val)

            # For DNN early stopping, split inner-train further into train/es-val
            tr2_idx, es_idx = train_test_split(
                np.arange(len(y_in_tr)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_in_tr
            )
            X2_es_tr = X2_in_tr_s[tr2_idx]
            y2_es_tr = y_in_tr[tr2_idx]
            X2_es_val = X2_in_tr_s[es_idx]
            y2_es_val = y_in_tr[es_idx]

            ch2 = train_channel2_dnn(X2_es_tr, y2_es_tr, X2_es_val, y2_es_val)
            oof_ch2[in_val] = channel2_predict_proba(ch2, X2_in_val_s)

            print(f"  Inner fold {inner_fold}/{n_splits} done")

        # Train fusion on OOF preds (outer-train only, leakage-safe)
        X_fused_oof = np.column_stack([oof_ch1, oof_ch2])
        fusion = train_fusion_model(X_fused_oof, y_tr)

        # ---- Refit base models on full outer-train, then infer outer-test probs
        # Channel 1 refit
        sc1_full = StandardScaler().fit(X1_tr)
        X1_tr_s = sc1_full.transform(X1_tr)
        X1_te_s = sc1_full.transform(X1_te)
        ch1_full = train_channel1_ensemble(X1_tr_s, y_tr)
        ch1_te_prob = channel1_predict_proba(ch1_full, X1_te_s)

        # Channel 2 refit with hold-out val from outer-train (early stopping)
        tr2_idx, es_idx = train_test_split(
            np.arange(len(y_tr)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
        )
        sc2_full = StandardScaler().fit(X2_tr[tr2_idx])
        X2_tr_s = sc2_full.transform(X2_tr)
        X2_te_s = sc2_full.transform(X2_te)

        X2_es_tr = X2_tr_s[tr2_idx]
        y2_es_tr = y_tr[tr2_idx]
        X2_es_val = X2_tr_s[es_idx]
        y2_es_val = y_tr[es_idx]

        ch2_full = train_channel2_dnn(X2_es_tr, y2_es_tr, X2_es_val, y2_es_val)
        ch2_te_prob = channel2_predict_proba(ch2_full, X2_te_s)

        # Fusion inference on outer-test
        X_te_fused = np.column_stack([ch1_te_prob, ch2_te_prob])
        y_te_prob = fusion_predict_proba(fusion, X_te_fused)

        m = compute_metrics(y_te, y_te_prob)
        fold_metrics.append(m)

        print(
            f"Outer fold {outer_fold} metrics: "
            f"ACC={m['acc']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}"
        )

    # Aggregate
    def agg(key: str):
        vals = np.array([fm[key] for fm in fold_metrics], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {
        "acc_mean": agg("acc")[0], "acc_std": agg("acc")[1],
        "precision_mean": agg("precision")[0], "precision_std": agg("precision")[1],
        "recall_mean": agg("recall")[0], "recall_std": agg("recall")[1],
        "f1_mean": agg("f1")[0], "f1_std": agg("f1")[1],
        "auc_mean": agg("auc")[0], "auc_std": agg("auc")[1],
    }
    return summary


def main():
    set_seed(RANDOM_STATE)

    # Load dataset 
    df = pd.read_csv('data/HeartCT.csv')

    if RUN_CV:
        summary = run_leakage_safe_cv(df, n_splits=N_SPLITS)
        print("\n===== 10-FOLD CV SUMMARY (Leakage-Safe Fusion) =====")
        print(f"ACC:  {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
        print(f"Prec: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        print(f"Rec:  {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        print(f"F1:   {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"AUC:  {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    else:
        m = run_single_split(df)
        print("\n===== SINGLE SPLIT TEST METRICS =====")
        print(f"ACC:  {m['acc']:.4f}")
        print(f"Prec: {m['precision']:.4f}")
        print(f"Rec:  {m['recall']:.4f}")
        print(f"F1:   {m['f1']:.4f}")
        print(f"AUC:  {m['auc']:.4f}")


if __name__ == "__main__":
    main()

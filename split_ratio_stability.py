# Dual-Channel Heart Disease Diagnosis — Split-Ratio Stability (70/30 + 90/10)
#
# This script extends the leakage-safe pipeline with *repeated stratified hold-out*
# evaluations for alternative split ratios, as requested for the Statistical Analysis.
#
# - Channel 1: (XGBoost + SVM) soft-voting ensemble
# - Channel 2: DNN with simple attention
# - Fusion: FusionNN trained on training-set base-model probabilities (hold-out setting)
#
# Notes on leakage-safety (hold-out setting):
#   - StandardScaler parameters are fit on the training partition only.
#   - All model fitting is done using training data only.
#   - Early stopping validation is drawn only from the training partition.

import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED_START = 42
N_REPEATS = 30                 
PRINT_EACH_REPEAT = True       # set False for quieter logs

SPLITS_TO_RUN = [
    ("70/30", 0.30),
    ("90/10", 0.10),
]

# DNN training knobs (same defaults as CV script)
DNN_MAX_EPOCHS = 300
DNN_BATCH_SIZE = 32
DNN_LR = 0.01
DNN_WEIGHT_DECAY = 1e-4
DNN_HIDDEN_UNITS = 256

# Fusion training knobs
FUSION_EPOCHS = 300
FUSION_LR = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
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


# =========================
# MODELS
# =========================
class Attention(nn.Module):
    """Retained to match your manuscript implementation."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = x @ self.attention_weights          # [B]
        scores = scores.unsqueeze(-1)                # [B, 1]
        scores = torch.softmax(scores, dim=1)        # [B, 1] (single column)
        scores = scores.squeeze(-1)                  # [B]
        return x * scores.unsqueeze(-1)              # [B, D]


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
# TRAIN / PRED HELPERS
# =========================
def train_channel1_ensemble(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> VotingClassifier:
    model_xgb = xgb.XGBClassifier(
        learning_rate=0.25,
        max_depth=3,
        n_estimators=500,
        subsample=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model_svm = SVC(C=100, gamma=100, kernel="rbf", probability=True)
    ensemble = VotingClassifier(
        estimators=[("xgb", model_xgb), ("svm", model_svm)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def channel1_predict_proba(model: VotingClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def train_channel2_dnn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> DNNWithAttention:
    model = DNNWithAttention(input_dim=X_train.shape[1], n_units=DNN_HIDDEN_UNITS).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=DNN_LR, weight_decay=DNN_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=20)

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

    for _epoch in range(DNN_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
    return model(xb).squeeze().detach().cpu().numpy()


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
    return fusion(xb).squeeze().detach().cpu().numpy()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "ACC": accuracy_score(y_true, y_pred),
        "P": precision_score(y_true, y_pred, zero_division=0),
        "R": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    return out


# =========================
# ONE HOLD-OUT RUN
# =========================
def run_one_holdout(df: pd.DataFrame, test_size: float, seed: int) -> dict:
    features_channel_1 = ["age", "sex", "chol", "trestbps", "fbs"]
    features_channel_2 = ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    target = "target"

    X1 = df[features_channel_1].values
    X2 = df[features_channel_2].values
    y = df[target].values.astype(int)

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    X1_tr, X1_te = X1[idx_tr], X1[idx_te]
    X2_tr, X2_te = X2[idx_tr], X2[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # ---- Channel 1 (train-only scaling)
    sc1 = StandardScaler().fit(X1_tr)
    X1_tr_s = sc1.transform(X1_tr)
    X1_te_s = sc1.transform(X1_te)

    ch1 = train_channel1_ensemble(X1_tr_s, y_tr, seed=seed)
    ch1_tr_prob = channel1_predict_proba(ch1, X1_tr_s)
    ch1_te_prob = channel1_predict_proba(ch1, X1_te_s)

    # ---- Channel 2 (train-only val split, train-only scaling)
    idx_tr2, idx_val2 = train_test_split(
        np.arange(len(y_tr)), test_size=0.2, random_state=seed, stratify=y_tr
    )
    X2_es_tr = X2_tr[idx_tr2]
    X2_es_val = X2_tr[idx_val2]
    y2_es_tr = y_tr[idx_tr2]
    y2_es_val = y_tr[idx_val2]

    sc2 = StandardScaler().fit(X2_es_tr)
    X2_es_tr_s = sc2.transform(X2_es_tr)
    X2_es_val_s = sc2.transform(X2_es_val)
    X2_tr_s = sc2.transform(X2_tr)
    X2_te_s = sc2.transform(X2_te)

    ch2 = train_channel2_dnn(X2_es_tr_s, y2_es_tr, X2_es_val_s, y2_es_val)
    ch2_tr_prob = channel2_predict_proba(ch2, X2_tr_s)
    ch2_te_prob = channel2_predict_proba(ch2, X2_te_s)

    # ---- Fusion (trained on training-set base probs)
    X_fused_tr = np.column_stack([ch1_tr_prob, ch2_tr_prob])
    X_fused_te = np.column_stack([ch1_te_prob, ch2_te_prob])

    fusion = train_fusion_model(X_fused_tr, y_tr)
    y_te_prob = fusion_predict_proba(fusion, X_fused_te)
    return compute_metrics(y_te, y_te_prob)


# =========================
# REPEATED HOLD-OUT
# =========================
def summarize_runs(rows: list[dict]) -> pd.DataFrame:
    """Return mean, std, and 95% CI for the mean (t-distribution)."""
    dfm = pd.DataFrame(rows)
    n = len(dfm)

    # t-critical (fallback to 1.96 if scipy is unavailable)
    try:
        from scipy.stats import t
        tcrit = float(t.ppf(0.975, df=n - 1))
    except Exception:
        tcrit = 1.96

    out = []
    for col in ["ACC", "P", "R", "F1", "AUC"]:
        mean = float(dfm[col].mean())
        std = float(dfm[col].std(ddof=1))
        se = std / np.sqrt(n) if n > 0 else float("nan")
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
        out.append({
            "Metric": col,
            "mean": mean,
            "std": std,
            "95%_CI_low": ci_low,
            "95%_CI_high": ci_high,
        })
    return pd.DataFrame(out)


def run_repeated_holdout(df: pd.DataFrame, test_size: float, seed_start: int, n_repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    for i in range(n_repeats):
        seed = seed_start + i
        set_seed(seed)
        m = run_one_holdout(df, test_size=test_size, seed=seed)
        runs.append(m)
        if PRINT_EACH_REPEAT:
            print(
                f"[Repeat {i+1}/{n_repeats}] seed={seed} | "
                f"ACC={m['ACC']:.4f}, P={m['P']:.4f}, R={m['R']:.4f}, F1={m['F1']:.4f}, AUC={m['AUC']:.4f}"
            )

    summary = summarize_runs(runs)
    summary_pct = summary.copy()
    for c in ["mean", "std", "95%_CI_low", "95%_CI_high"]:
        summary_pct[c] = 100.0 * summary_pct[c]
    return summary, summary_pct


def make_tiny_table(label: str, summary_pct: pd.DataFrame) -> pd.DataFrame:
    """Compact table: ACC/F1/AUC only (most reviewers expect these three)."""
    keep = summary_pct[summary_pct["Metric"].isin(["ACC", "F1", "AUC"])].copy()
    keep["Split"] = label
    keep = keep[["Split", "Metric", "mean", "std", "95%_CI_low", "95%_CI_high"]]
    # Format-friendly rounding
    return keep.round({"mean": 2, "std": 2, "95%_CI_low": 2, "95%_CI_high": 2})


def make_tiny_table_wide(tiny_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form tiny table into a 1-row-per-split table for easy pasting."""
    def fmt(row):
        return f"{row['mean']:.2f} ± {row['std']:.2f}  (95% CI: {row['95%_CI_low']:.2f}–{row['95%_CI_high']:.2f})"

    wide_rows = []
    for split in tiny_long["Split"].unique():
        sub = tiny_long[tiny_long["Split"] == split]
        d = {"Split": split}
        for metric in ["ACC", "F1", "AUC"]:
            r = sub[sub["Metric"] == metric].iloc[0]
            d[metric] = fmt(r)
        wide_rows.append(d)
    return pd.DataFrame(wide_rows)


def main() -> None:
    print(f"Device: {DEVICE}")

    # Load dataset 
    df = pd.read_csv('data/HeartCT.csv')

    tiny_tables = []
    for label, test_size in SPLITS_TO_RUN:
        print(f"\n==== Repeated Hold-out: {label} (test_size={test_size}) ====")
        summary, summary_pct = run_repeated_holdout(df, test_size, seed_start=SEED_START, n_repeats=N_REPEATS)

        print(f"\n---- Summary (mean ± std, 95% CI for mean) [{label}] ----")
        print(summary)

        print(f"\n---- Same summary in % [{label}] ----")
        print(summary_pct)

        tiny_tables.append(make_tiny_table(label, summary_pct))

    tiny = pd.concat(tiny_tables, ignore_index=True)
    print("\n==== Tiny Stability Table (%, ACC/F1/AUC) ====")
    print(tiny)

    tiny_wide = make_tiny_table_wide(tiny)
    print("\n==== Tiny Stability Table (wide, ready to paste) ====")
    print(tiny_wide)

    # Save for direct paste into manuscript (optional)
    tiny.to_csv("split_ratio_stability_tiny.csv", index=False)
    tiny_wide.to_csv("split_ratio_stability_tiny_wide.csv", index=False)
    print("\nSaved: split_ratio_stability_tiny.csv")
    print("Saved: split_ratio_stability_tiny_wide.csv")


if __name__ == "__main__":
    main()

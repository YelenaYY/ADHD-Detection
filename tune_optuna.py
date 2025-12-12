# tune_optuna.py
"""
Hyperparameter tuning for SpatioTemporalADHDModel using Optuna + SQLite storage.

- Saves all trials to SQLite DB: adhd_optuna.db
- Study name: "adhd_spatiotemporal_transformer"
- Can be visualized with optuna-dashboard.

Assumes:
  - dataloader.py defines: build_datasets(site_config=...)
  - transformer.py defines: SpatioTemporalADHDModel
  - Per-subject ROI files: data/processed/roi_tensor/sub-*.npy
  - Label CSVs: data/NYU_labels.csv, data/NEURO_labels.csv, etc.
"""

from pathlib import Path
import numpy as np
import optuna
from optuna.pruners import MedianPruner

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from dataloader import build_datasets
from transformer import SpatioTemporalADHDModel


# -----------------------------
# Global config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust these paths if needed
ROOT = Path(__file__).resolve().parent
SITE_CONFIG = {
    "NYU": {
        "roi_npy": str(ROOT / "data" / "processed" / "roi_tensor" / "NYU_roi_timeseries.npy"),
        "label_csv": str(ROOT / "data" / "NYU_labels.csv"),
    },
    "NEURO": {
        "roi_npy": str(ROOT / "data" / "processed" / "roi_tensor" / "NEURO_roi_timeseries.npy"),
        "label_csv": str(ROOT / "data" / "NEURO_labels.csv"),
    },
    "OHSU": {
        "roi_npy": str(ROOT / "data" / "processed" / "roi_tensor" / "OHSU_roi_timeseries.npy"),
        "label_csv": str(ROOT / "data" / "OHSU_labels.csv"),
    },
    "PEKING": {
        "roi_npy": str(ROOT / "data" / "processed" / "roi_tensor" / "PEKING_roi_timeseries.npy"),
        "label_csv": str(ROOT / "data" / "PEKING_labels.csv"),
    },
}

N_EPOCHS = 30
EXPECTED_ROIS = 90

# Optuna storage config
DB_PATH = "adhd_optuna.db"   # file will be created in current directory
STUDY_NAME = "adhd_spatiotemporal_transformer"
STORAGE_URL = f"sqlite:///{DB_PATH}"


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(DEVICE)                 # (B, T, R)
        y = y.to(DEVICE).float()         # (B,)

        optimizer.zero_grad()
        logits = model(x).squeeze(-1)    # (B,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            total_correct += (preds.cpu() == y.long().cpu()).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * x.size(0)
            total_correct += (preds.cpu() == y.long().cpu()).sum().item()
            total_samples += x.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # AUC only if both classes present
    if len(np.unique(all_labels)) == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = np.nan
    else:
        auc = np.nan

    return avg_loss, acc, auc


# -----------------------------
# Optuna objective
# -----------------------------
def objective_bce(trial: optuna.Trial) -> float:
    set_seed(42)

    # ---- Hyperparameter search space ----
    d_model = trial.suggest_categorical("d_model", [8, 16, 32])
    temporal_hidden_channels = trial.suggest_categorical(
        "temporal_hidden_channels", [16, 32, 64]
    )

    # nhead must divide d_model
    possible_heads = [2, 4, 8]
    possible_heads = [h for h in possible_heads if h <= d_model and d_model % h == 0]
    if not possible_heads:
        possible_heads = [1]
    transformer_nhead = trial.suggest_categorical("transformer_nhead", possible_heads)

    transformer_num_layers = trial.suggest_int("transformer_num_layers", 1, 3)
    transformer_dim_feedforward = trial.suggest_categorical(
        "transformer_dim_feedforward", [64, 128, 256]
    )
    transformer_dropout = trial.suggest_float("transformer_dropout", 0.1, 0.5)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    window_T = trial.suggest_categorical("window_T", [None, 120, 176])

    # ---- Datasets & loaders (deterministic splits) ----
    train_ds, val_ds, test_ds = build_datasets(
        site_config=SITE_CONFIG,
        label_col="label",
        binary_label=True,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        normalize=True,
        window_T=window_T,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ---- Model, loss, optimizer ----
    model = SpatioTemporalADHDModel(
        n_rois=EXPECTED_ROIS,
        d_model=d_model,
        temporal_hidden_channels=temporal_hidden_channels,
        temporal_kernel_size=3,  # unused for GRU but kept for API
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_dropout=transformer_dropout,
        num_classes=1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_score = -np.inf

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)

        # Objective: maximize AUC when available, otherwise accuracy
        score = val_auc if not np.isnan(val_auc) else val_acc
        best_val_score = max(best_val_score, score)

        trial.report(score, epoch)

        if trial.should_prune():
            print(f"[Trial {trial.number}] pruned at epoch {epoch}")
            raise optuna.TrialPruned()

        print(
            f"[Trial {trial.number:03d}] Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.3f}, auc {val_auc:.3f}"
        )

    return best_val_score


def objective(trial: optuna.Trial) -> float:
    set_seed(42)

    # ---- Hyperparameter search space ----
    d_model = trial.suggest_categorical("d_model", [8, 16, 32])
    temporal_hidden_channels = trial.suggest_categorical(
        "temporal_hidden_channels", [8, 16, 32, 64]
    )

    possible_heads = [2, 4, 8]
    possible_heads = [h for h in possible_heads if h <= d_model and d_model % h == 0]
    if not possible_heads:
        possible_heads = [1]
    transformer_nhead = trial.suggest_categorical("transformer_nhead", possible_heads)

    transformer_num_layers = trial.suggest_int("transformer_num_layers", 1, 3)
    transformer_dim_feedforward = trial.suggest_categorical(
        "transformer_dim_feedforward", [8, 16, 32, 64, 128, 256]
    )
    transformer_dropout = trial.suggest_float("transformer_dropout", 0.1, 0.5)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    window_T = trial.suggest_categorical("window_T", [None, 120, 176, 261])

    # ---- Datasets & loaders ----
    train_ds, val_ds, test_ds = build_datasets(
        site_config=SITE_CONFIG,
        label_col="label",
        binary_label=True,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        normalize=True,
        window_T=window_T,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ---- Model ----
    model = SpatioTemporalADHDModel(
        n_rois=EXPECTED_ROIS,
        d_model=d_model,
        temporal_hidden_channels=temporal_hidden_channels,
        temporal_kernel_size=3,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_dropout=transformer_dropout,
        num_classes=1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_acc = -np.inf

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)

        # 👉 Optimization target: validation accuracy
        score = val_acc
        best_val_acc = max(best_val_acc, val_acc)

        # Report to Optuna so pruning & dashboard see ACC
        trial.report(score, epoch)

        if trial.should_prune():
            print(f"[Trial {trial.number}] pruned at epoch {epoch}")
            raise optuna.TrialPruned()

        print(
            f"[Trial {trial.number:03d}] Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.3f}, auc {val_auc:.3f}"
        )

    # 👉 Return best validation accuracy over epochs
    return best_val_acc


# -----------------------------
# Main: create / load study + optimize
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    print(f"[info] Using Optuna SQLite storage: {STORAGE_URL}")
    print(f"[info] Study name: {STUDY_NAME}")

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,   # ← reuse existing study in the same DB
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=100, timeout=None)

    print("\n=== Optuna finished ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value (AUC / ACC): {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

# main.py
"""
Training driver for SpatioTemporal ADHD Transformer on ADHD-200 ROI time series.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import build_dataloaders
from transformer import SpatioTemporalADHDModel


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)           # (B, T, R)
        y = y.to(device).float()   # (B,)

        optimizer.zero_grad()
        logits = model(x).squeeze(-1)    # (B,)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        running_correct += (preds == y.long()).sum().item()
        n_samples += x.size(0)

    avg_loss = running_loss / n_samples
    avg_acc = running_correct / n_samples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()

            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            running_correct += (preds == y.long()).sum().item()
            n_samples += x.size(0)

    avg_loss = running_loss / n_samples
    avg_acc = running_correct / n_samples
    return avg_loss, avg_acc


def main():
    # ---- Config -------------------------------------------------------------
    site_config = {
        "NYU": {
            "roi_npy": "/home/yelena/Desktop/Git/ADHD-Detection/data/processed/roi_tensor/NYU_roi_timeseries.npy",
            "label_csv": "/home/yelena/Desktop/Git/ADHD-Detection/data/NYU_labels.csv",
        },
        "NEURO": {
            "roi_npy": "/home/yelena/Desktop/Git/ADHD-Detection/data/processed/roi_tensor/NEURO_roi_timeseries.npy",
            "label_csv": "/home/yelena/Desktop/Git/ADHD-Detection/data/NEURO_labels.csv",
        },
        "OHSU": {
            "roi_npy": "/home/yelena/Desktop/Git/ADHD-Detection/data/processed/roi_tensor/OHSU_roi_timeseries.npy",
            "label_csv": "/home/yelena/Desktop/Git/ADHD-Detection/data/OHSU_labels.csv",
        },
        "PEKING": {
            "roi_npy": "/home/yelena/Desktop/Git/ADHD-Detection/data/processed/roi_tensor/PEKING_roi_timeseries.npy",
            "label_csv": "/home/yelena/Desktop/Git/ADHD-Detection/data/PEKING_labels.csv",
        },
    }

    batch_size = 10
    num_epochs = 40
    lr = 0.001
    weight_decay = 1e-5

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        site_config=site_config,
        label_col="label",
        binary_label=True,
        time_first=True,     # (N, T, R)
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        normalize=True,
    )

    # ---- Model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalADHDModel(
        n_rois=90,
        d_model=64,
        temporal_hidden_channels=32,
        temporal_kernel_size=3,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_feedforward=128,
        transformer_dropout=0.1,
        num_classes=1,  # binary classification with BCEWithLogitsLoss
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training loop ------------------------------------------------------
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Simple early stopping-like tracking (you can extend this)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_spatiotemporal_transformer.pt")
            print("  [*] New best model saved.")

    # ---- Final test ---------------------------------------------------------
    print("\nEvaluating on test set with best saved model...")
    model.load_state_dict(torch.load("best_spatiotemporal_transformer.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

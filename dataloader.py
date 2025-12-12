# dataloader.py
"""
Dataloader utilities for ADHD fMRI ROI time-series.

- Loads per-site ROI tensors from .npy files (shape: N, T, R).
- Loads labels/phenotypes from CSVs.
- Combines multiple sites.
- Splits into train/val/test with stratification.
- Returns PyTorch DataLoaders.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class ROITimeSeriesArrayDataset(Dataset):
    """
    Simple Dataset wrapping numpy arrays X (N, T, R) and y (N,).
    Supports optional random cropping along time dimension.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_T: int | None = None):
        super().__init__()
        ...
        self.X = X
        self.y = y.astype(np.int64)
        self.window_T = window_T   # if None -> use full T

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]            # (T, R)
        y = self.y[idx]            # scalar

        if self.window_T is not None:
            T = x.shape[0]
            if self.window_T < T:
                # random contiguous crop
                start = np.random.randint(0, T - self.window_T + 1)
                x = x[start:start + self.window_T, :]   # (window_T, R)
            # if window_T >= T, just use full sequence (already padded earlier)

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def _load_site_arrays(
    roi_npy_path: str,
    label_csv_path: str,
    label_col: str = "label",
    binary_label: bool = True,
    time_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X (N, T, R) and y (N,) for a single site.
    Forces fixed time dimension T=176.
    """
    # Load ROI tensor
    X = np.load(roi_npy_path)  # (N, T, R) or (N, R, T)
    if X.ndim != 3:
        raise ValueError(f"{roi_npy_path} must be 3D (N, T, R) or (N, R, T), got {X.shape}")

    # Force (N, T, R)
    if time_first:
        X_ts = X                     # (N, T, R)
    else:
        X_ts = np.transpose(X, (0, 2, 1))

    # ------------------------------------------------------------------
    # 🔧 FIXED TIME DIMENSION (truncate/pad to T=176)
    # ------------------------------------------------------------------
    TARGET_T = 176
    N, T_orig, R = X_ts.shape
    X_fixed = np.zeros((N, TARGET_T, R), dtype=X_ts.dtype)

    for i in range(N):
        t = min(T_orig, TARGET_T)
        X_fixed[i, :t] = X_ts[i, :t]

    X_ts = X_fixed  # replace
    # ------------------------------------------------------------------

    # Load labels
    df = pd.read_csv(label_csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {label_csv_path}")

    df = df[~df[label_col].isna()].reset_index(drop=True)

    n_csv = len(df)
    n_npy = X_ts.shape[0]
    if n_csv != n_npy:
        min_n = min(n_csv, n_npy)
        print(
            f"[warn] Mismatch between CSV rows ({n_csv}) and NPY samples ({n_npy}) "
            f"for {label_csv_path}. Truncating both to {min_n}."
        )
        df = df.iloc[:min_n]
        X_ts = X_ts[:min_n]

    y = df[label_col].values.astype(np.int64)

    if binary_label:
        valid_mask = np.isin(y, [0, 1, 2, 3])
        if not valid_mask.all():
            n_bad = (~valid_mask).sum()
            print(f"[warn] Found {n_bad} rows with unexpected labels in {label_csv_path}. Dropping them.")
            X_ts = X_ts[valid_mask]
            y = y[valid_mask]

        y = (y > 0).astype(np.int64)

    return X_ts, y


def build_datasets(
    site_config: Dict[str, Dict[str, str]],
    label_col: str = "label",
    binary_label: bool = True,
    time_first: bool = True,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
    window_T: int | None = 180,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load all sites, combine, and split into train/val/test datasets.

    Parameters
    ----------
    site_config : dict
        Mapping from site name to dict with keys:
          - 'roi_npy': path to ROI .npy file
          - 'label_csv': path to labels CSV
    label_col : str, optional
        Name of label column in CSV.
    binary_label : bool, optional
        If True, map label>0 to 1.
    time_first : bool, optional
        If True, expects .npy as (N, T, R). If False, (N, R, T).
    val_size : float, optional
        Fraction of data to use as validation (of full dataset, not of train only).
    test_size : float, optional
        Fraction of data to use as test.
    random_state : int, optional
        Random seed for splits.

    Returns
    -------
    train_ds, val_ds, test_ds : Dataset
    """
    X_list = []
    y_list = []

    for site, cfg in site_config.items():
        print(f"[info] Loading site {site}")
        X_site, y_site = _load_site_arrays(
            roi_npy_path=cfg["roi_npy"],
            label_csv_path=cfg["label_csv"],
            label_col=label_col,
            binary_label=binary_label,
            time_first=time_first,
        )
        print(f"       X_site shape: {X_site.shape}, y_site shape: {y_site.shape}")
        X_list.append(X_site)
        y_list.append(y_site)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print(f"[info] Combined X shape: {X_all.shape}, y shape: {y_all.shape}")
    n_samples = X_all.shape[0]

    if normalize:
        # X_all: (N, T, R)
        mean = X_all.mean(axis=1, keepdims=True)        # (N, 1, R)
        std = X_all.std(axis=1, keepdims=True)          # (N, 1, R)
        X_all = (X_all - mean) / (std + 1e-6)
        print("[info] Applied per-subject per-ROI z-scoring along time.")

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        stratify=y_all,
        random_state=random_state,
    )

    # Compute val fraction relative to remaining data
    val_frac_of_temp = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac_of_temp,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"[info] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    train_ds = ROITimeSeriesArrayDataset(X_train, y_train, window_T=window_T)
    # For val/test, we usually want deterministic behavior, so either:
    # - no crop, or
    # - center crop. For simplicity, just no crop:
    val_ds = ROITimeSeriesArrayDataset(X_val, y_val, window_T=None)
    test_ds = ROITimeSeriesArrayDataset(X_test, y_test, window_T=None)

    return train_ds, val_ds, test_ds


def build_dataloaders(
    site_config: Dict[str, Dict[str, str]],
    label_col: str = "label",
    binary_label: bool = True,
    time_first: bool = True,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    normalize: bool = True,
    window_T: int | None = 180,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience wrapper: build train/val/test datasets and wrap them in DataLoaders.
    """
    train_ds, val_ds, test_ds = build_datasets(
        site_config=site_config,
        label_col=label_col,
        binary_label=binary_label,
        time_first=time_first,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        normalize=normalize,
        window_T=window_T,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

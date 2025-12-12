## ADHD-Detection

Spatio-temporal Transformer for ADHD-200 fMRI ROI time-series classification.

### Environment

1) Create and activate the Conda env:

```bash
conda env create -f environment.yml
conda activate adhd-detection
```

2) Optional: verify PyTorch + CUDA

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Data

Preprocessing is performed in a separate repository. The necessary dataframe is hosted on GitHub, and the preprocessed ROI tensors are generated there. This repository only consumes those artifacts.

#### Quickstart (prepared dataframes)

No download step is required if you are using the prepared files:

1) Ensure the following files exist in this repo:
   - Labels:
     - `data/NYU_labels.csv`
     - `data/NEURO_labels.csv`
     - `data/OHSU_labels.csv`
     - `data/PEKING_labels.csv`
   - ROI tensors:
     - `data/processed/roi_tensor/NYU_roi_timeseries.npy`
     - `data/processed/roi_tensor/NEURO_roi_timeseries.npy`
     - `data/processed/roi_tensor/OHSU_roi_timeseries.npy`
     - `data/processed/roi_tensor/PEKING_roi_timeseries.npy`

2) Then run training directly:

```bash
python main.py
```

That’s it—no dataset download is necessary.

Place files as follows:
- Phenotypic/label dataframe(s) under `data/`
  - If you have a single consolidated dataframe, export per-site CSVs named:
    - `data/NYU_labels.csv`
    - `data/NEURO_labels.csv`
    - `data/OHSU_labels.csv`
    - `data/PEKING_labels.csv`
  - Each CSV must include a `label` column. The row order must match the corresponding ROI tensor order.
- Preprocessed ROI time-series NumPy arrays under `data/processed/roi_tensor/`:
  - `NYU_roi_timeseries.npy`
  - `NEURO_roi_timeseries.npy`
  - `OHSU_roi_timeseries.npy`
  - `PEKING_roi_timeseries.npy`

Shapes should be `(N, T, R)` where:
- `N` = number of subjects
- `T` = time points (the loader will pad/truncate to 176)
- `R` = number of ROIs (expected 90)

Note:
- If you're pulling the dataframe from GitHub (e.g., release assets or a data folder), download it and save under `data/` as described above.
- Ensure that for each site, CSV rows and ROI array samples align one-to-one and in the same order.


### Training

Paths are repository-relative by default (no absolute paths). Ensure files are placed under `data/` as above, then run:

```bash
python main.py
```

Artifacts:
- Best model weights: `best_spatiotemporal_transformer.pt`

### Hyperparameter Tuning (Optuna)

`tune_optuna.py` runs an Optuna study stored in `adhd_optuna.db` (SQLite). Update paths in `SITE_CONFIG` if needed, then run:

```bash
python tune_optuna.py
```

Inspect the study interactively:

```bash
optuna-dashboard sqlite:///adhd_optuna.db
```

### Dataloader behavior

- Forces time dimension to a fixed length \(T = 176\) by truncation/padding.
- Optionally applies per-subject, per-ROI z-scoring along time.
- Supports optional random temporal cropping during training via `window_T`.
- Binary labels are derived from the provided `label` column: values > 0 are mapped to 1.
- If CSV rows and NPY samples mismatch, the loader will truncate to the smaller count and print a warning.



### Repository layout (key files)

- `dataloader.py`: Loads ROI arrays and labels, builds `Dataset` and `DataLoader`s.
- `transformer.py`: Temporal 1D-CNN encoder + ROI Transformer classifier.
- `main.py`: Training loop with early-best checkpointing.
- `tune_optuna.py`: Optuna study for hyperparameter search; logs to SQLite.

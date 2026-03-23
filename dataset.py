# dataset.py
# PyTorch Dataset and DataLoader for the MIT-BIH Arrhythmia Database.
#
# Each record is loaded, z-score normalised, then split into overlapping
# sliding windows.  The train / validation split is done at the *record*
# level (not sample level) to prevent temporal data leakage.
#
# GenAI assistance: used to draft the sliding-window logic; logic was verified
# against the wfdb documentation and manual inspection of sample outputs.

import os
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN,
    STRIDE, BATCH_SIZE, TRAIN_VAL_SPLIT, SEED,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_record_names(data_folder: str) -> list[str]:
    """Return sorted list of record stems (e.g. ['100', '101', ...]) found
    in *data_folder* by looking for .dat files."""
    names = sorted(
        set(f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat"))
    )
    return names


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    """Sliding-window dataset over a set of MIT-BIH records.

    Each item is a pair (x, y):
      x : FloatTensor of shape (INPUT_LEN,  1) — past ECG signal (model input)
      y : FloatTensor of shape (FORECAST_LEN,)  — future ECG values (target)

    The channel dimension in x is 1 so the LSTM receives shape
    (batch, seq_len, 1) directly.
    """

    def __init__(
        self,
        records: list[str],
        data_folder: str = DATA_FOLDER,
        input_len: int = INPUT_LEN,
        forecast_len: int = FORECAST_LEN,
        stride: int = STRIDE,
    ) -> None:
        self.input_len = input_len
        self.forecast_len = forecast_len
        self._windows: list[tuple[np.ndarray, np.ndarray]] = []

        for name in records:
            record_path = os.path.join(data_folder, name)
            record = wfdb.rdrecord(record_path)
            # Use lead 0 (MLII in most MIT-BIH records)
            signal = record.p_signal[:, 0].astype(np.float32)

            # Per-record z-score normalisation
            mu, sigma = signal.mean(), signal.std()
            signal = (signal - mu) / (sigma + 1e-8)

            # Sliding windows
            window = input_len + forecast_len
            for start in range(0, len(signal) - window + 1, stride):
                x = signal[start : start + input_len]
                y = signal[start + input_len : start + window]
                self._windows.append((x, y))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        x, y = self._windows[idx]
        x_t = torch.from_numpy(x).unsqueeze(-1)  # (input_len, 1)
        y_t = torch.from_numpy(y)                 # (forecast_len,)
        return x_t, y_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_folder: str = DATA_FOLDER,
    input_len: int = INPUT_LEN,
    forecast_len: int = FORECAST_LEN,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    split: float = TRAIN_VAL_SPLIT,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader).

    The split is record-level: a randomly chosen *split* fraction of records
    go to training and the remainder to validation.  num_workers=0 is used
    for Windows compatibility.
    """
    rng = np.random.default_rng(seed)
    records = get_record_names(data_folder)
    records = list(rng.permutation(records))  # shuffle deterministically

    n_train = int(len(records) * split)
    train_records = records[:n_train]
    val_records   = records[n_train:]

    train_ds = ECGDataset(train_records, data_folder, input_len, forecast_len, stride)
    val_ds   = ECGDataset(val_records,   data_folder, input_len, forecast_len, stride)

    print(f"[dataset] Train: {len(train_records)} records | {len(train_ds):,} windows")
    print(f"[dataset] Val:   {len(val_records)} records | {len(val_ds):,} windows")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader

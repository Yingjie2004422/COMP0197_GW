# dataset.py
# PyTorch Dataset and DataLoader for the MIT-BIH Arrhythmia Database.
#
# Each item returned is a 4-tuple:
#   x_signal : FloatTensor (input_len, 1)   — normalised past ECG signal
#   x_annot  : LongTensor  (input_len,)     — per-timestep beat label (0/1/2)
#   y_signal : FloatTensor (forecast_len,)  — normalised future ECG signal
#   y_risk   : FloatTensor scalar           — 1.0 if an abnormal beat occurs
#                                             in the forecast window, else 0.0
#
# Beat labels
# -----------
#   0 — no beat annotation at this sample
#   1 — normal beat  (N, L, R, B, e, j, n)
#   2 — abnormal beat (V, A, a, J, S, E, F, f, /, Q, !)
#
# The train/val split is performed at the *record* level so that no
# temporal information leaks from training into validation windows.
#
# GenAI assistance: used to draft the annotation-loading and window-indexing
# logic; verified against wfdb documentation and manual inspection of outputs.

import os
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN,
    STRIDE, BATCH_SIZE, TRAIN_VAL_SPLIT, SEED,
)

# Beat symbols taken from the AAMI / MIT-BIH annotation standard.
NORMAL_BEATS   = frozenset(["N", "L", "R", "B", "e", "j", "n"])
ABNORMAL_BEATS = frozenset(["V", "A", "a", "J", "S", "E", "F", "f", "/", "Q", "!"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_record_names(data_folder: str) -> list[str]:
    """Return sorted list of record stems found in *data_folder* (.dat files)."""
    return sorted(
        set(f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat"))
    )


def _load_record(data_folder: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one MIT-BIH record and return (signal, annot_array).

    signal      : float32 array of length T — z-score normalised lead-I values
    annot_array : int64   array of length T — 0/1/2 beat label at each sample
    """
    path   = os.path.join(data_folder, name)
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, 0].astype(np.float32)

    # Per-record z-score normalisation
    mu, sigma = signal.mean(), signal.std()
    signal = (signal - mu) / (sigma + 1e-8)

    # Build per-timestep annotation array (default 0 = no beat)
    annot_array = np.zeros(len(signal), dtype=np.int64)
    try:
        ann = wfdb.rdann(path, "atr")
        for sample_idx, symbol in zip(ann.sample, ann.symbol):
            if sample_idx >= len(signal):
                continue
            if symbol in NORMAL_BEATS:
                annot_array[sample_idx] = 1
            elif symbol in ABNORMAL_BEATS:
                annot_array[sample_idx] = 2
            # Other symbols (rhythm markers, noise, etc.) stay 0
    except Exception:
        # A small number of records lack a usable .atr file; treat as no-beat.
        pass

    return signal, annot_array


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    """Sliding-window dataset over a collection of MIT-BIH records.

    Signals and annotation arrays are held in memory per record; individual
    windows are sliced on demand in __getitem__ to avoid storing O(N_windows)
    copies of overlapping arrays.
    """

    def __init__(
        self,
        records: list[str],
        data_folder: str = DATA_FOLDER,
        input_len: int   = INPUT_LEN,
        forecast_len: int = FORECAST_LEN,
        stride: int      = STRIDE,
    ) -> None:
        self.input_len    = input_len
        self.forecast_len = forecast_len

        # _data[i] = (signal_array, annot_array) for record i
        self._data: list[tuple[np.ndarray, np.ndarray]] = []
        # _index[j] = (record_idx, window_start) for window j
        self._index: list[tuple[int, int]] = []

        for r_idx, name in enumerate(records):
            signal, annot = _load_record(data_folder, name)
            self._data.append((signal, annot))

            window = input_len + forecast_len
            for start in range(0, len(signal) - window + 1, stride):
                self._index.append((r_idx, start))

        # Compute class-imbalance weight for arrhythmia risk (BCE pos_weight).
        # pos_weight = #negative_windows / #positive_windows.
        # Vectorised per record: group window starts by record, then use a
        # cumsum to check forecast ranges in one numpy operation per record.
        from collections import defaultdict
        by_record: dict = defaultdict(list)
        for ri, start in self._index:
            by_record[ri].append(start)

        pos = 0
        for r_idx, (_, annot) in enumerate(self._data):
            starts = np.array(by_record[r_idx], dtype=np.int32)
            if len(starts) == 0:
                continue
            # prefix sum: cumsum[i] = number of abnormal beats in annot[0:i]
            prefix = np.concatenate([[0], np.cumsum(annot == 2)])
            fc_starts = starts + input_len
            fc_ends   = fc_starts + forecast_len
            pos += int(np.sum(prefix[fc_ends] - prefix[fc_starts] > 0))

        neg = len(self._index) - pos
        self.pos_weight = float(neg) / max(pos, 1)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        r_idx, start = self._index[idx]
        signal, annot = self._data[r_idx]

        end = start + self.input_len

        x_sig = signal[start:end]                            # (input_len,)
        x_ann = annot[start:end]                             # (input_len,)
        y_sig = signal[end : end + self.forecast_len]        # (forecast_len,)

        # Binary arrhythmia risk: 1.0 if any abnormal beat in forecast window
        y_risk = float(2 in annot[end : end + self.forecast_len])

        return (
            torch.from_numpy(x_sig).unsqueeze(-1),           # (input_len, 1)  float32
            torch.from_numpy(x_ann),                         # (input_len,)    int64
            torch.from_numpy(y_sig),                         # (forecast_len,) float32
            torch.tensor(y_risk, dtype=torch.float32),       # scalar
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_folder: str  = DATA_FOLDER,
    input_len: int    = INPUT_LEN,
    forecast_len: int = FORECAST_LEN,
    stride: int       = STRIDE,
    batch_size: int   = BATCH_SIZE,
    split: float      = TRAIN_VAL_SPLIT,
    seed: int         = SEED,
) -> tuple[DataLoader, DataLoader, float]:
    """Build and return (train_loader, val_loader, pos_weight).

    pos_weight is the BCE imbalance weight computed from the training split.
    The split is record-level.  num_workers=0 for Windows compatibility.
    """
    rng     = np.random.default_rng(seed)
    records = list(rng.permutation(get_record_names(data_folder)))

    n_train       = int(len(records) * split)
    train_records = records[:n_train]
    val_records   = records[n_train:]

    train_ds = ECGDataset(train_records, data_folder, input_len, forecast_len, stride)
    val_ds   = ECGDataset(val_records,   data_folder, input_len, forecast_len, stride)

    print(f"[dataset] Train: {len(train_records)} records | {len(train_ds):,} windows "
          f"| arrhythmia pos_weight: {train_ds.pos_weight:.2f}")
    print(f"[dataset] Val:   {len(val_records)} records | {len(val_ds):,} windows")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, train_ds.pos_weight

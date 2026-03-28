# dataset.py
# PyTorch Dataset and DataLoader for the MIT-BIH Arrhythmia Database.
#
# Each item returned is a 7-tuple:
#   x_signal    : FloatTensor (input_len, 2)   — normalised past ECG signal (2 leads)
#   x_annot     : LongTensor  (input_len,)     — per-timestep beat label (0/1/2)
#   x_feat      : FloatTensor (input_len, 4)   — [RR, dRR, peak_amp, dAmp]
#   y_signal    : FloatTensor (forecast_len,)  — normalised future ECG signal (lead 0)
#   y_risk      : FloatTensor scalar           — 1.0 if an abnormal beat occurs
#                                               in the forecast window, else 0.0
#   y_beat_type : LongTensor  scalar           — dominant beat type in forecast
#                                               window (0-4 encoding)
#   x_hrv       : FloatTensor (3,)             — [SDNN, RMSSD, pNN50] from input window
#
# Beat labels (annot_arr / x_annot)
# -----------------------------------
#   0 — no beat annotation at this sample
#   1 — normal beat  (N, L, R, B, e, j, n)
#   2 — abnormal beat (V, A, a, J, S, E, F, f, /, Q, !)
#
# Beat type labels (beat_type_arr / y_beat_type)
# ------------------------------------------------
#   0 — no beat
#   1 — normal  (N, L, R, B, e, j, n)
#   2 — PVC     (V, E, !)
#   3 — APB     (A, a, J, S)
#   4 — other abnormal (F, f, /, Q)
#
# Feature channels (x_feat)
# --------------------------
#   ch 0  RR       = time (seconds) since previous beat, normalised per record
#   ch 1  dRR      = change in RR between consecutive beats, normalised per record
#   ch 2  peak_amp = ECG amplitude at each beat position, normalised per record
#   ch 3  dAmp     = change in peak amplitude between consecutive beats, normalised
# All four channels are forward-filled from beat positions.
#
# The train/val split is at the *record* level to prevent temporal leakage.
# The val set is further split 20%/80% into cal_records (conformal calibration)
# and val_records (held-out evaluation).
#
# GenAI assistance: used to draft the annotation-loading, RR computation, and
# window-indexing logic; extended with beat morphology features, conformal
# split, dual-lead input, and HRV features by the team; verified against wfdb
# documentation and manual inspection of sample outputs.

import os
from collections import defaultdict

import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN,
    STRIDE, BATCH_SIZE, TRAIN_VAL_SPLIT, SEED, SAMPLING_RATE,
    USE_OVERSAMPLING,
)

NORMAL_BEATS   = frozenset(["N", "L", "R", "B", "e", "j", "n"])
PVC_BEATS      = frozenset(["V", "E", "!"])
APB_BEATS      = frozenset(["A", "a", "J", "S"])
OTHER_ABNORMAL = frozenset(["F", "f", "/", "Q"])
# Union of all beats that trigger an arrhythmia label (annot == 2)
ABNORMAL_BEATS = PVC_BEATS | APB_BEATS | OTHER_ABNORMAL


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ffill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a 1D float array.

    Positions before the first valid value are filled with that first value
    (nearest-neighbour fill). Returns a float32 array of the same length.
    """
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0:
        return np.zeros(len(arr), dtype=np.float32)
    # For each position, find the index of the most recent valid entry
    ptr = np.searchsorted(valid, np.arange(len(arr)), side="right") - 1
    ptr = np.clip(ptr, 0, len(valid) - 1)
    out = np.where(np.isnan(arr), arr[valid[ptr]], arr)
    return out.astype(np.float32)


def get_record_names(data_folder: str) -> list[str]:
    """Return sorted list of record stems (e.g. ['100', '101', ...])."""
    return sorted(
        set(f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat"))
    )


# ---------------------------------------------------------------------------
# Record loader
# ---------------------------------------------------------------------------

def _load_record(
    data_folder: str, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one MIT-BIH record.

    Returns
    -------
    signal       : float32 (T, 2)  — z-score normalised dual-lead ECG signal
    annot_arr    : int64   (T,)    — beat label 0/1/2 at each sample
    beat_type_arr: int64   (T,)    — beat type 0-4 at each sample
    feat_arr     : float32 (T, 4)  — [RR_norm, dRR_norm, amp_norm, dAmp_norm]
                                     all forward-filled from beat positions
    """
    path   = os.path.join(data_folder, name)
    record = wfdb.rdrecord(path)

    # Load both leads; normalise each independently
    raw = record.p_signal[:, :min(2, record.p_signal.shape[1])].astype(np.float32)
    if raw.ndim == 1 or raw.shape[1] < 2:
        raw = np.stack([raw.ravel(), raw.ravel()], axis=1)
    mu0, s0 = raw[:, 0].mean(), raw[:, 0].std() + 1e-8
    mu1, s1 = raw[:, 1].mean(), raw[:, 1].std() + 1e-8
    signal = np.stack([(raw[:, 0] - mu0) / s0, (raw[:, 1] - mu1) / s1], axis=1)  # (T, 2)

    T = signal.shape[0]

    # --- Annotation and beat type arrays ---
    annot_arr     = np.zeros(T, dtype=np.int64)
    beat_type_arr = np.zeros(T, dtype=np.int64)
    beat_pos      = []   # sample indices of all labelled beats

    try:
        ann = wfdb.rdann(path, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            if s >= T:
                continue
            if sym in NORMAL_BEATS:
                annot_arr[s]     = 1
                beat_type_arr[s] = 1
                beat_pos.append(s)
            elif sym in PVC_BEATS:
                annot_arr[s]     = 2
                beat_type_arr[s] = 2
                beat_pos.append(s)
            elif sym in APB_BEATS:
                annot_arr[s]     = 2
                beat_type_arr[s] = 3
                beat_pos.append(s)
            elif sym in OTHER_ABNORMAL:
                annot_arr[s]     = 2
                beat_type_arr[s] = 4
                beat_pos.append(s)
    except Exception:
        pass  # records without a usable .atr file are treated as annotation-free

    # --- RR interval features + beat amplitude features ---
    rr_raw   = np.full(T, np.nan, dtype=np.float32)
    drr_raw  = np.full(T, np.nan, dtype=np.float32)
    amp_raw  = np.full(T, np.nan, dtype=np.float32)
    damp_raw = np.full(T, np.nan, dtype=np.float32)

    if len(beat_pos) >= 2:
        bs  = np.array(beat_pos)                       # sorted beat positions

        # RR intervals (seconds)
        rr  = np.diff(bs) / SAMPLING_RATE              # (n-1,)
        rr  = np.concatenate([[rr[0]], rr])            # (n,)  prepend first RR
        drr = np.concatenate([[0.0], np.diff(rr)])     # (n,)  dRR = change in RR

        # Per-record normalisation using beat-level statistics
        rr_mean,  rr_std  = rr.mean(),  rr.std()  + 1e-8
        drr_mean, drr_std = drr.mean(), drr.std() + 1e-8

        # Beat amplitude = lead-0 ECG signal value at each beat position
        amp  = signal[bs, 0].astype(np.float32)        # (n,)
        damp = np.concatenate([[0.0], np.diff(amp)])   # (n,)

        amp_mean,  amp_std  = amp.mean(),  amp.std()  + 1e-8
        damp_mean, damp_std = damp.mean(), damp.std() + 1e-8

        for i, pos in enumerate(bs):
            rr_raw[pos]   = (rr[i]   - rr_mean)   / rr_std
            drr_raw[pos]  = (drr[i]  - drr_mean)  / drr_std
            amp_raw[pos]  = (amp[i]  - amp_mean)  / amp_std
            damp_raw[pos] = (damp[i] - damp_mean) / damp_std

    # Forward-fill sparse beat values across every sample
    feat_arr = np.stack(
        [_ffill(rr_raw), _ffill(drr_raw), _ffill(amp_raw), _ffill(damp_raw)],
        axis=1,
    )  # (T, 4)

    return signal, annot_arr, beat_type_arr, feat_arr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    """Sliding-window dataset over a collection of MIT-BIH records.

    Raw signal, annotation, beat-type, and feature arrays are held in memory
    per record; individual windows are sliced on demand in __getitem__ to
    avoid storing O(N_windows) copies of heavily overlapping arrays.

    Attributes
    ----------
    normal_indices     : list of window indices where forecast window has no arrhythmia
    arrhythmia_indices : list of window indices where forecast window has arrhythmia
    pos_weight         : #negative_windows / #positive_windows (for BCE loss)
    """

    def __init__(
        self,
        records:      list[str],
        data_folder:  str = DATA_FOLDER,
        input_len:    int = INPUT_LEN,
        forecast_len: int = FORECAST_LEN,
        stride:       int = STRIDE,
    ) -> None:
        self.input_len    = input_len
        self.forecast_len = forecast_len

        # _data[r] = (signal, annot, beat_type, feat_arr) for record r
        self._data:  list[tuple] = []
        # _index[j]  = (record_idx, window_start) for window j
        self._index: list[tuple] = []

        for r_idx, name in enumerate(records):
            signal, annot, beat_type, feat_arr = _load_record(data_folder, name)
            self._data.append((signal, annot, beat_type, feat_arr))
            window = input_len + forecast_len
            for start in range(0, signal.shape[0] - window + 1, stride):
                self._index.append((r_idx, start))

        # --- Compute BCE pos_weight and per-window arrhythmia flags ---
        by_record: dict = defaultdict(list)
        for ri, s in self._index:
            by_record[ri].append(s)

        self.normal_indices: list[int]     = []
        self.arrhythmia_indices: list[int] = []

        pos = 0
        global_idx = 0
        for r_idx, (_, annot, _bt, _f) in enumerate(self._data):
            starts = np.array(by_record[r_idx], dtype=np.int32)
            if len(starts) == 0:
                continue
            # Prefix sum over abnormal-beat indicator for O(n) range queries
            prefix    = np.concatenate([[0], np.cumsum(annot == 2)])
            fc_starts = starts + input_len
            fc_ends   = fc_starts + forecast_len
            has_arrhythmia = (prefix[fc_ends] - prefix[fc_starts]) > 0

            for j, flag in enumerate(has_arrhythmia):
                # global_idx tracks position in self._index across all records
                win_global = self._record_start_idx(r_idx, by_record) + j
                if flag:
                    self.arrhythmia_indices.append(win_global)
                    pos += 1
                else:
                    self.normal_indices.append(win_global)

        neg = len(self._index) - pos
        self.pos_weight = float(neg) / max(pos, 1)

    def _record_start_idx(self, r_idx: int, by_record: dict) -> int:
        """Return the global starting index for record r_idx in self._index."""
        offset = 0
        for i in range(r_idx):
            offset += len(by_record[i])
        return offset

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        r_idx, start = self._index[idx]
        signal, annot, beat_type, feat_arr = self._data[r_idx]

        end    = start + self.input_len
        x_sig  = signal[start:end]                         # (input_len, 2)
        x_ann  = annot[start:end]                          # (input_len,)
        x_feat = feat_arr[start:end]                       # (input_len, 4)
        y_sig  = signal[end:end + self.forecast_len, 0]    # (forecast_len,)  lead 0 only

        # Binary arrhythmia risk label
        y_risk = float(2 in annot[end : end + self.forecast_len])

        # Dominant beat type in forecast window (most frequent non-zero type,
        # or 0 if no beats present)
        fc_types = beat_type[end : end + self.forecast_len]
        nonzero  = fc_types[fc_types > 0]
        if len(nonzero) > 0:
            counts = np.bincount(nonzero, minlength=5)
            y_beat_type = int(counts[1:].argmax()) + 1   # offset by 1 (skip 0)
        else:
            y_beat_type = 0

        # HRV features from beats in input window [start, end)
        beat_pos_win = np.where(annot[start:end] > 0)[0]
        if len(beat_pos_win) >= 2:
            rr = np.diff(beat_pos_win) / SAMPLING_RATE   # in seconds
            sdnn  = float(rr.std())
            rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr) > 1 else 0.0
            pnn50 = float(np.mean(np.abs(np.diff(rr)) > 0.050)) if len(rr) > 1 else 0.0
        else:
            sdnn = rmssd = pnn50 = 0.0
        x_hrv = torch.tensor([sdnn, rmssd, pnn50], dtype=torch.float32)

        return (
            torch.from_numpy(x_sig),                        # (input_len, 2)   float32
            torch.from_numpy(x_ann),                        # (input_len,)     int64
            torch.from_numpy(x_feat),                       # (input_len, 4)   float32
            torch.from_numpy(y_sig),                        # (forecast_len,)  float32
            torch.tensor(y_risk,       dtype=torch.float32),  # scalar float32
            torch.tensor(y_beat_type,  dtype=torch.int64),    # scalar int64
            x_hrv,                                          # (3,)             float32
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_folder:  str   = DATA_FOLDER,
    input_len:    int   = INPUT_LEN,
    forecast_len: int   = FORECAST_LEN,
    stride:       int   = STRIDE,
    batch_size:   int   = BATCH_SIZE,
    split:        float = TRAIN_VAL_SPLIT,
    seed:         int   = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, float]:
    """Build and return (train_loader, val_loader, cal_loader, pos_weight).

    Splits
    ------
    - Records are shuffled deterministically then split train/val at the
      record level (no patient data shared between sets).
    - val_records are further divided: first 20% -> cal_records (conformal
      calibration), remaining 80% -> val_records (held-out evaluation).

    Oversampling
    ------------
    If USE_OVERSAMPLING is True, a WeightedRandomSampler is used for the
    train_loader so that arrhythmia windows are sampled at pos_weight× the
    rate of normal windows.

    num_workers=0 is used for Windows/macOS compatibility.
    """
    rng     = np.random.default_rng(seed)
    records = get_record_names(data_folder)
    records = list(rng.permutation(records))

    n_train       = int(len(records) * split)
    train_records = records[:n_train]
    all_val_recs  = records[n_train:]

    # Split val records: 20% cal / 80% val
    n_cal       = max(1, int(len(all_val_recs) * 0.20))
    cal_records = all_val_recs[:n_cal]
    val_records = all_val_recs[n_cal:]

    train_ds = ECGDataset(train_records, data_folder, input_len, forecast_len, stride)
    val_ds   = ECGDataset(val_records,   data_folder, input_len, forecast_len, stride)
    cal_ds   = ECGDataset(cal_records,   data_folder, input_len, forecast_len, stride)

    print(
        f"[dataset] Train: {len(train_records)} records | "
        f"{len(train_ds):,} windows | pos_weight: {train_ds.pos_weight:.2f}"
    )
    print(
        f"[dataset] Val:   {len(val_records)} records | "
        f"{len(val_ds):,} windows"
    )
    print(
        f"[dataset] Cal:   {len(cal_records)} records | "
        f"{len(cal_ds):,} windows  (conformal calibration)"
    )

    # --- Train loader: optionally oversample arrhythmia windows ---
    if USE_OVERSAMPLING and len(train_ds.arrhythmia_indices) > 0:
        weights = np.ones(len(train_ds), dtype=np.float64)
        for idx in train_ds.arrhythmia_indices:
            weights[idx] = train_ds.pos_weight
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, cal_loader, train_ds.pos_weight

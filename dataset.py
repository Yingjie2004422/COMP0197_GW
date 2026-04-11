# dataset.py
# PyTorch Dataset and DataLoader for the MIT-BIH Arrhythmia Database.
#
# Each item returned is a 7-tuple:
#   x_signal    : FloatTensor (input_len, 2)   — normalised past ECG signal (2 leads)
#   x_annot     : LongTensor  (input_len,)     — per-timestep beat label (0/1/2)
#   x_feat      : FloatTensor (input_len, 8)   — 8 feature channels (see below)
#   y_signal    : FloatTensor (forecast_len,)  — normalised future ECG signal (lead 0)
#   y_risk      : FloatTensor scalar           — 1.0 if an abnormal beat occurs
#                                               in the forecast window, else 0.0
#   y_beat_type : LongTensor  scalar           — dominant beat type in forecast
#                                               window (0-4 encoding)
#   x_hrv       : FloatTensor (3,)             — [SDNN, RMSSD, pNN50] from input window
#
# Feature channels (x_feat) — 8 total
# ------------------------------------
#   ch 0  RR        = time since previous beat (s), normalised per record (forward-filled)
#   ch 1  dRR       = change in RR, normalised per record (forward-filled)
#   ch 2  peak_amp  = ECG amplitude at beat position, normalised per record (forward-filled)
#   ch 3  dAmp      = change in peak amplitude, normalised per record (forward-filled)
#   ch 4  QRS_width = QRS complex width in samples, normalised per record (forward-filled)
#   ch 5  low_band  = relative signal power 0.5–5 Hz (broadcast across window)
#   ch 6  mid_band  = relative signal power 5–20 Hz (broadcast across window)
#   ch 7  high_band = relative signal power 20–50 Hz (broadcast across window)
#
# Normalisation: robust (median / IQR) instead of z-score to resist artifact spikes.
#
# Signal quality filter: windows with spike > 6 IQR-units or near-flatline (std < 0.05)
# are removed from the index at construction time.
#
# Sampler: beat-type balanced WeightedRandomSampler ensures each epoch sees
# proportional representation of Normal / PVC / APB / Other beat types.
#
# GenAI assistance: used to draft annotation-loading, RR computation, window-indexing,
# QRS width, frequency band features, and quality filtering; verified against wfdb docs.

import os
from collections import defaultdict

import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN,
    STRIDE, BATCH_SIZE, TRAIN_VAL_SPLIT, SEED, SAMPLING_RATE,
    USE_OVERSAMPLING,
)

NORMAL_BEATS   = frozenset(["N", "L", "R", "B", "e", "j", "n"])
PVC_BEATS      = frozenset(["V", "E", "!"])
APB_BEATS      = frozenset(["A", "a", "J", "S"])
OTHER_ABNORMAL = frozenset(["F", "f", "/", "Q"])
ABNORMAL_BEATS = PVC_BEATS | APB_BEATS | OTHER_ABNORMAL


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ffill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a 1D float array."""
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0:
        return np.zeros(len(arr), dtype=np.float32)
    ptr = np.searchsorted(valid, np.arange(len(arr)), side="right") - 1
    ptr = np.clip(ptr, 0, len(valid) - 1)
    out = np.where(np.isnan(arr), arr[valid[ptr]], arr)
    return out.astype(np.float32)


def _robust_norm(x: np.ndarray) -> np.ndarray:
    """Normalise a 1-D signal using median and IQR (robust to spike artifacts)."""
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return ((x - med) / (iqr + 1e-8)).astype(np.float32)


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
    signal        : float32 (T, 2)  — robust-normalised dual-lead ECG
    annot_arr     : int64   (T,)    — beat label 0/1/2
    beat_type_arr : int64   (T,)    — beat type 0-4
    feat_arr      : float32 (T, 5)  — [RR, dRR, peak_amp, dAmp, QRS_width]
                                      forward-filled from beat positions
    """
    path   = os.path.join(data_folder, name)
    record = wfdb.rdrecord(path)

    # Load both leads; robust-normalise each independently
    raw = record.p_signal[:, :min(2, record.p_signal.shape[1])].astype(np.float32)
    if raw.ndim == 1 or raw.shape[1] < 2:
        raw = np.stack([raw.ravel(), raw.ravel()], axis=1)
    signal = np.stack([_robust_norm(raw[:, 0]), _robust_norm(raw[:, 1])], axis=1)
    T = signal.shape[0]

    # --- Annotation and beat type arrays ---
    annot_arr     = np.zeros(T, dtype=np.int64)
    beat_type_arr = np.zeros(T, dtype=np.int64)
    beat_pos      = []

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
        pass

    # --- Beat-level features: RR, dRR, peak_amp, dAmp, QRS_width ---
    rr_raw    = np.full(T, np.nan, dtype=np.float32)
    drr_raw   = np.full(T, np.nan, dtype=np.float32)
    amp_raw   = np.full(T, np.nan, dtype=np.float32)
    damp_raw  = np.full(T, np.nan, dtype=np.float32)
    qrsw_raw  = np.full(T, np.nan, dtype=np.float32)

    if len(beat_pos) >= 2:
        bs = np.array(beat_pos)

        # RR intervals
        rr  = np.diff(bs) / SAMPLING_RATE
        rr  = np.concatenate([[rr[0]], rr])
        drr = np.concatenate([[0.0], np.diff(rr)])

        rr_mean,  rr_std  = rr.mean(),  rr.std()  + 1e-8
        drr_mean, drr_std = drr.mean(), drr.std() + 1e-8

        # Peak amplitude (lead 0)
        amp  = signal[bs, 0].astype(np.float32)
        damp = np.concatenate([[0.0], np.diff(amp)])

        amp_mean,  amp_std  = amp.mean(),  amp.std()  + 1e-8
        damp_mean, damp_std = damp.mean(), damp.std() + 1e-8

        # QRS width: number of samples within ±40 of R peak where
        # |signal| > 30% of peak magnitude (robust proxy for QRS duration)
        qrs_widths = []
        HALF_WIN = 40
        for pos in bs:
            left  = max(0, pos - HALF_WIN)
            right = min(T - 1, pos + HALF_WIN)
            seg   = signal[left:right + 1, 0]
            peak_val = float(signal[pos, 0])
            if abs(peak_val) > 0.05:
                threshold = 0.30 * abs(peak_val)
                above = np.where(np.abs(seg) > threshold)[0]
                w = int(above[-1] - above[0]) if len(above) >= 2 else 5
            else:
                w = 5
            qrs_widths.append(float(w))
        qrs_widths = np.array(qrs_widths, dtype=np.float32)
        qrsw_mean, qrsw_std = qrs_widths.mean(), qrs_widths.std() + 1e-8

        for i, pos in enumerate(bs):
            rr_raw[pos]   = (rr[i]         - rr_mean)   / rr_std
            drr_raw[pos]  = (drr[i]        - drr_mean)  / drr_std
            amp_raw[pos]  = (amp[i]        - amp_mean)  / amp_std
            damp_raw[pos] = (damp[i]       - damp_mean) / damp_std
            qrsw_raw[pos] = (qrs_widths[i] - qrsw_mean) / qrsw_std

    feat_arr = np.stack(
        [_ffill(rr_raw), _ffill(drr_raw), _ffill(amp_raw),
         _ffill(damp_raw), _ffill(qrsw_raw)],
        axis=1,
    )  # (T, 5)

    return signal, annot_arr, beat_type_arr, feat_arr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    """Sliding-window dataset over MIT-BIH records.

    Quality filtering removes windows with spike artifacts (|signal| > 6) or
    near-flatlines (std < 0.05) at construction time.

    Attributes
    ----------
    normal_indices     : global window indices with no forecast-window arrhythmia
    arrhythmia_indices : global window indices with forecast-window arrhythmia
    window_beat_types  : dominant beat type (0-4) per window (for balanced sampler)
    pos_weight         : neg / pos count ratio (for BCE loss)
    """

    def __init__(
        self,
        records:      list[str],
        data_folder:  str = DATA_FOLDER,
        input_len:    int = INPUT_LEN,
        forecast_len: int = FORECAST_LEN,
        stride:       int = STRIDE,
        jitter:       int = 0,
    ) -> None:
        self.jitter       = jitter
        self.input_len    = input_len
        self.forecast_len = forecast_len

        self._data:  list[tuple] = []
        self._index: list[tuple] = []

        n_filtered = 0
        for r_idx, name in enumerate(records):
            signal, annot, beat_type, feat_arr = _load_record(data_folder, name)
            self._data.append((signal, annot, beat_type, feat_arr))
            window = input_len + forecast_len
            for start in range(0, signal.shape[0] - window + 1, stride):
                end = start + input_len
                seg = signal[start:end + forecast_len, 0]
                # Quality filter: reject spike artifacts and flatlines
                if np.max(np.abs(seg)) > 6.0 or np.std(seg) < 0.05:
                    n_filtered += 1
                    continue
                self._index.append((r_idx, start))

        if n_filtered > 0:
            print(f"[dataset] Quality filter removed {n_filtered:,} windows")

        # --- Per-window arrhythmia flags and beat types ---
        by_record: dict = defaultdict(list)
        for ri, s in self._index:
            by_record[ri].append(s)

        self.normal_indices:     list[int] = []
        self.arrhythmia_indices: list[int] = []
        self.window_beat_types:  list[int] = []

        pos = 0
        for r_idx, (_, annot, beat_type, _f) in enumerate(self._data):
            starts = np.array(by_record[r_idx], dtype=np.int32)
            if len(starts) == 0:
                continue
            prefix    = np.concatenate([[0], np.cumsum(annot == 2)])
            fc_starts = starts + input_len
            fc_ends   = fc_starts + forecast_len
            has_arrhythmia = (prefix[fc_ends] - prefix[fc_starts]) > 0

            offset = self._record_start_idx(r_idx, by_record)
            for j, (flag, fc_s, fc_e) in enumerate(
                zip(has_arrhythmia, fc_starts, fc_ends)
            ):
                win_global = offset + j
                # Dominant beat type in forecast window
                fc_types = beat_type[fc_s:fc_e]
                nonzero  = fc_types[fc_types > 0]
                if len(nonzero) > 0:
                    counts = np.bincount(nonzero, minlength=5)
                    dom_type = int(counts[1:].argmax()) + 1
                else:
                    dom_type = 0
                self.window_beat_types.append(dom_type)

                if flag:
                    self.arrhythmia_indices.append(win_global)
                    pos += 1
                else:
                    self.normal_indices.append(win_global)

        neg = len(self._index) - pos
        self.pos_weight = float(neg) / max(pos, 1)

    def _record_start_idx(self, r_idx: int, by_record: dict) -> int:
        offset = 0
        for i in range(r_idx):
            offset += len(by_record[i])
        return offset

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        r_idx, start = self._index[idx]
        signal, annot, beat_type, feat_arr = self._data[r_idx]

        # Random window jitter
        if self.jitter > 0:
            shift     = np.random.randint(-self.jitter, self.jitter + 1)
            max_start = signal.shape[0] - self.input_len - self.forecast_len
            start     = int(np.clip(start + shift, 0, max_start))

        end    = start + self.input_len
        x_sig  = signal[start:end]                          # (input_len, 2)
        x_ann  = annot[start:end]                           # (input_len,)
        x_feat = feat_arr[start:end]                        # (input_len, 5)
        y_sig  = signal[end:end + self.forecast_len, 0]     # (forecast_len,)

        # --- Frequency band features (computed per window, broadcast) ---
        # Relative power in three clinically meaningful ECG bands:
        #   Low  (0.5–5 Hz)  : P-wave, baseline, slow rhythm changes
        #   Mid  (5–20 Hz)   : QRS onset/offset, T-wave
        #   High (20–50 Hz)  : QRS peak morphology, high-freq noise
        lead0   = x_sig[:, 0].astype(np.float64)
        fft_mag = np.abs(np.fft.rfft(lead0)) ** 2
        freqs   = np.fft.rfftfreq(self.input_len, d=1.0 / SAMPLING_RATE)
        low_p   = fft_mag[(freqs >= 0.5) & (freqs <  5.0)].sum()
        mid_p   = fft_mag[(freqs >= 5.0) & (freqs < 20.0)].sum()
        high_p  = fft_mag[(freqs >= 20.0) & (freqs < 50.0)].sum()
        total   = low_p + mid_p + high_p + 1e-12
        freq_feats = np.array(
            [low_p / total, mid_p / total, high_p / total], dtype=np.float32
        )
        # Broadcast scalar band powers to every timestep → (input_len, 3)
        freq_arr = np.broadcast_to(freq_feats, (self.input_len, 3)).copy()

        # Concatenate: (input_len, 5) + (input_len, 3) = (input_len, 8)
        x_feat = np.concatenate([x_feat, freq_arr], axis=1)

        # --- Binary arrhythmia risk label ---
        y_risk = float(2 in annot[end:end + self.forecast_len])

        # --- Dominant beat type in forecast window ---
        fc_types = beat_type[end:end + self.forecast_len]
        nonzero  = fc_types[fc_types > 0]
        if len(nonzero) > 0:
            counts      = np.bincount(nonzero, minlength=5)
            y_beat_type = int(counts[1:].argmax()) + 1
        else:
            y_beat_type = 0

        # --- HRV features from input window ---
        beat_pos_win = np.where(annot[start:end] > 0)[0]
        if len(beat_pos_win) >= 2:
            rr    = np.diff(beat_pos_win) / SAMPLING_RATE
            sdnn  = float(rr.std())
            rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
            pnn50 = float(np.mean(np.abs(np.diff(rr)) > 0.050)) if len(rr) > 1 else 0.0
        else:
            sdnn = rmssd = pnn50 = 0.0
        x_hrv = torch.tensor([sdnn, rmssd, pnn50], dtype=torch.float32)

        return (
            torch.from_numpy(x_sig),                           # (input_len, 2)
            torch.from_numpy(x_ann),                           # (input_len,)
            torch.from_numpy(x_feat),                          # (input_len, 8)
            torch.from_numpy(y_sig),                           # (forecast_len,)
            torch.tensor(y_risk,      dtype=torch.float32),    # scalar
            torch.tensor(y_beat_type, dtype=torch.int64),      # scalar
            x_hrv,                                             # (3,)
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

    Sampler: beat-type balanced WeightedRandomSampler — each beat type
    (Normal/PVC/APB/Other) is sampled at a rate inversely proportional to
    its frequency, ensuring the model trains on rare arrhythmia types evenly.
    """
    rng     = np.random.default_rng(seed)
    records = get_record_names(data_folder)
    records = list(rng.permutation(records))

    n_train       = int(len(records) * split)
    train_records = records[:n_train]
    all_val_recs  = records[n_train:]

    n_cal       = max(1, int(len(all_val_recs) * 0.20))
    cal_records = all_val_recs[:n_cal]
    val_records = all_val_recs[n_cal:]

    train_ds = ECGDataset(train_records, data_folder, input_len, forecast_len, stride, jitter=45)
    val_ds   = ECGDataset(val_records,   data_folder, input_len, forecast_len, stride, jitter=0)
    cal_ds   = ECGDataset(cal_records,   data_folder, input_len, forecast_len, stride, jitter=0)

    print(
        f"[dataset] Train: {len(train_records)} records | "
        f"{len(train_ds):,} windows | pos_weight: {train_ds.pos_weight:.2f}"
    )
    print(f"[dataset] Val:   {len(val_records)} records | {len(val_ds):,} windows")
    print(f"[dataset] Cal:   {len(cal_records)} records | {len(cal_ds):,} windows")

    # --- Beat-type balanced sampler ---
    # Weight each window inversely by its dominant beat type's frequency.
    # This replaces the binary arrhythmia oversampler with a finer-grained version
    # that balances Normal / PVC / APB / Other separately.
    if USE_OVERSAMPLING and len(train_ds.arrhythmia_indices) > 0:
        types  = np.array(train_ds.window_beat_types, dtype=np.int32)
        counts = np.bincount(types, minlength=5).astype(np.float64)
        # Inverse-frequency weights per type; type 0 (no beat) treated as Normal
        type_w = np.zeros(5, dtype=np.float64)
        for t in range(5):
            type_w[t] = 1.0 / counts[t] if counts[t] > 0 else 0.0
        weights = type_w[types]
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(weights),
            num_samples = len(train_ds),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=False,
        )
        type_names = {0:"no-beat", 1:"normal", 2:"PVC", 3:"APB", 4:"other"}
        for t in range(5):
            if counts[t] > 0:
                print(f"[dataset]   {type_names[t]:<8}: {int(counts[t]):>7,} windows  "
                      f"weight={type_w[t]:.6f}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, cal_loader, train_ds.pos_weight

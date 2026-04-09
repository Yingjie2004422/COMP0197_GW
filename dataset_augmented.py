"""
Subtype-aware and heterogeneity-aware dataset pipeline for MIT-BIH ECG forecasting.

What is new compared with the original dataset pipeline
-------------------------------------------------------
The original dataset pipeline mainly did three things:
1. loaded dual-lead ECG and per-sample beat labels,
2. computed beat-derived features such as RR, dRR, amplitude, and dAmp,
3. forward-filled those sparse beat features across all samples and used
   a mostly binary positive/negative oversampling strategy.

This updated pipeline changes the data handling in three important ways.

1) Window labels are no longer only "arrhythmia" vs "no arrhythmia"
   ----------------------------------------------------------------
   Each input/forecast pair is assigned a transition subtype:
   - normal: no abnormal beat in the input or forecast window
   - pre_event: no abnormal beat in the input, but abnormal beat in forecast
   - persistent_abnormal: abnormal beat appears in both input and forecast
   - recovery: abnormal beat appears in input, but not in forecast

   This makes the sampling policy more clinically meaningful because it
   separates onset prediction from persistent abnormal-state recognition.

2) Sampling is balanced over subtype x record-group buckets
   --------------------------------------------------------
   Instead of only oversampling windows with future arrhythmia, the dataset
   groups windows by both:
   - window subtype, and
   - record phenotype / domain group
     (mostly_normal, mixed, persistent_abnormal, paced_or_bbb)

   Sample weights are then computed at the bucket level so that rare but
   informative combinations, such as pre_event windows in mixed records,
   are seen more often during training.

3) Sparse beat-derived features are no longer represented only by forward-fill
   ---------------------------------------------------------------------------
   The original code exposed dense beat-derived channels after forward-filling
   sparse values across every timestep. That representation is still kept as
   x_feat, but two additional signals are now provided:

   - x_feat_mask:
       indicates where RR/dRR/amplitude/dAmp values were truly observed at
       beat positions instead of merely propagated by forward-fill.

   - x_beat_event:
       sparse beat-event representation with channels such as beat presence,
       normal/abnormal flag, RR, dRR, amplitude, and dAmp at actual beat
       locations.

   This allows future models to distinguish "observed beat information" from
   "imputed between-beat context".

How augmentation is updated compared with before
------------------------------------------------
The old training pipeline applied only signal-level augmentations such as:
- Gaussian noise,
- global amplitude scaling,
- baseline wander.

This updated pipeline keeps that idea but makes augmentation more consistent
with the structure of ECG and sparse beat features:

- signal-level augmentation:
    mild per-lead scaling, low-amplitude noise, baseline wander, optional
    partial lead attenuation

- sparse-feature augmentation:
    small perturbations applied only at observed feature positions
    (using x_feat_mask), and optional random dropping of a subset of sparse
    beat events from x_beat_event

- stochastic per draw:
    the same window can receive a different augmented version every time it
    is sampled, which is especially important when WeightedRandomSampler
    revisits rare windows many times

How to adapt the model architecture to actually use the new augmentation-aware inputs
-------------------------------------------------------------------------------------
The current dataset returns more information than the original model consumes.
If the model still only takes:
    x_signal, x_annot, x_feat, x_hrv
then it benefits from the updated sampling and signal augmentation, but it does
not explicitly use x_feat_mask or x_beat_event.

A minimal augmentation-aware architecture would do one of the following:

1. Mask-aware dense feature fusion
   Concatenate x_feat and x_feat_mask before the feature encoder so the model
   knows which values are directly observed and which are forward-filled.

2. Separate beat-event encoder
   Encode x_beat_event with a small MLP / 1D conv / recurrent branch and fuse
   it with the waveform encoder. This preserves event sparsity instead of
   forcing the model to infer beat timing only from dense channels.

3. Multi-branch fusion
   Use separate encoders for:
   - waveform (x_signal),
   - per-sample annotation stream (x_annot),
   - dense beat features + mask (x_feat, x_feat_mask),
   - sparse beat-event stream (x_beat_event),
   then fuse these representations before forecasting and risk prediction.

4. Attention over beat events
   Let the decoder or risk head attend specifically to sparse event positions,
   which is useful for onset prediction and rhythm transition modeling.

In short:
- this dataset already improves what the model sees and how often it sees it,
- but architectural changes are still required if the model is to directly
  consume the new mask and beat-event representations.
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from config import (
        DATA_FOLDER, INPUT_LEN, FORECAST_LEN,
        STRIDE, BATCH_SIZE, TRAIN_VAL_SPLIT, SEED, SAMPLING_RATE,
    )
except Exception:
    DATA_FOLDER = "./mitdb"
    INPUT_LEN = 360
    FORECAST_LEN = 180
    STRIDE = 180
    BATCH_SIZE = 64
    TRAIN_VAL_SPLIT = 0.8
    SEED = 42
    SAMPLING_RATE = 360


# Keep beat groups close to the original code, but treat conduction / paced beats
# separately at the record-group level to better reflect dataset heterogeneity.
NORMAL_BEATS = frozenset(["N", "B", "e", "j", "n"])
BBB_BEATS = frozenset(["L", "R"])
PVC_BEATS = frozenset(["V", "E", "!"])
APB_BEATS = frozenset(["A", "a", "J", "S", "x"])
PACED_BEATS = frozenset(["/", "f"])
OTHER_ABNORMAL = frozenset(["F", "Q"])
ABNORMAL_BEATS = BBB_BEATS | PVC_BEATS | APB_BEATS | PACED_BEATS | OTHER_ABNORMAL
ALL_BEAT_SYMBOLS = NORMAL_BEATS | ABNORMAL_BEATS

WINDOW_SUBTYPES = {
    "normal": 0,
    "pre_event": 1,
    "persistent_abnormal": 2,
    "recovery": 3,
}
SUBTYPE_NAMES = {v: k for k, v in WINDOW_SUBTYPES.items()}

RECORD_GROUPS = {
    "mostly_normal": 0,
    "mixed": 1,
    "persistent_abnormal": 2,
    "paced_or_bbb": 3,
}
RECORD_GROUP_NAMES = {v: k for k, v in RECORD_GROUPS.items()}


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def _ffill_with_mask(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = arr.astype(np.float32)
    mask = (~np.isnan(arr)).astype(np.float32)
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return np.zeros(len(arr), dtype=np.float32), np.zeros(len(arr), dtype=np.float32)
    ptr = np.searchsorted(valid, np.arange(len(arr)), side="right") - 1
    ptr = np.clip(ptr, 0, len(valid) - 1)
    filled = np.where(np.isnan(arr), arr[valid[ptr]], arr).astype(np.float32)
    return filled, mask


def get_record_names(data_folder: str) -> List[str]:
    recs = sorted(set(f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat")))
    if not recs:
        raise FileNotFoundError(f"No .dat records found in {data_folder}")
    return recs


def _simple_signal_augment(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = signal.copy().astype(np.float32)
    T, C = x.shape

    if rng.random() < 0.7:
        scale = rng.uniform(0.95, 1.05, size=(1, C)).astype(np.float32)
        x *= scale

    if rng.random() < 0.6:
        x += rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)

    if rng.random() < 0.5:
        t = np.arange(T, dtype=np.float32)
        freq = rng.uniform(0.05, 0.4)
        phase = rng.uniform(0.0, 2 * np.pi)
        amp = rng.uniform(0.0, 0.03)
        bw = (amp * np.sin(2 * np.pi * freq * t / SAMPLING_RATE + phase)).astype(np.float32)
        x += bw[:, None]

    if C > 1 and rng.random() < 0.2:
        lead = int(rng.integers(0, C))
        x[:, lead] *= rng.uniform(0.0, 0.3)

    return x


def _augment_sparse_features(
    feat_dense: np.ndarray,
    feat_obs_mask: np.ndarray,
    beat_event: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    feat_dense = feat_dense.copy().astype(np.float32)
    beat_event = beat_event.copy().astype(np.float32)

    if rng.random() < 0.5:
        obs = feat_obs_mask > 0
        if obs.any():
            feat_dense[obs] += rng.normal(0.0, 0.03, size=feat_dense[obs].shape).astype(np.float32)

    if rng.random() < 0.25:
        event_positions = np.where(np.abs(beat_event[:, 0]) > 0)[0]
        if len(event_positions) > 0:
            drop_n = max(1, int(0.1 * len(event_positions)))
            drop_idx = rng.choice(event_positions, size=min(drop_n, len(event_positions)), replace=False)
            beat_event[drop_idx] = 0.0

    return feat_dense, beat_event


@dataclass
class RecordData:
    name: str
    signal: np.ndarray
    annot_arr: np.ndarray
    beat_type_arr: np.ndarray
    feat_dense: np.ndarray
    feat_obs_mask: np.ndarray
    beat_event: np.ndarray
    record_group: int
    meta: Dict[str, float]


def _classify_record_group(symbols: List[str], abnormal_ratio: float) -> int:
    counts = Counter(symbols)
    total = max(sum(counts.values()), 1)
    paced_fraction = (counts.get("/", 0) + counts.get("f", 0)) / total
    bbb_fraction = (counts.get("L", 0) + counts.get("R", 0)) / total

    # Keep paced/BBB as a distinct morphology/domain bucket, but only when truly dominant.
    if paced_fraction >= 0.5 or bbb_fraction >= 0.8:
        return RECORD_GROUPS["paced_or_bbb"]
    if abnormal_ratio >= 0.8:
        return RECORD_GROUPS["persistent_abnormal"]
    if abnormal_ratio <= 0.05:
        return RECORD_GROUPS["mostly_normal"]
    return RECORD_GROUPS["mixed"]


def _load_record(data_folder: str, name: str) -> RecordData:
    path = os.path.join(data_folder, name)
    record = wfdb.rdrecord(path)

    raw = record.p_signal[:, : min(2, record.p_signal.shape[1])].astype(np.float32)
    if raw.ndim == 1 or raw.shape[1] < 2:
        raw = np.stack([raw.ravel(), raw.ravel()], axis=1)
    signal = np.stack([_zscore(raw[:, 0]), _zscore(raw[:, 1])], axis=1)
    T = signal.shape[0]

    annot_arr = np.zeros(T, dtype=np.int64)
    beat_type_arr = np.zeros(T, dtype=np.int64)
    beat_pos: List[int] = []
    beat_symbols: List[str] = []

    try:
        ann = wfdb.rdann(path, "atr")
        for s, sym in zip(ann.sample, ann.symbol):
            if s >= T or sym not in ALL_BEAT_SYMBOLS:
                continue
            beat_pos.append(int(s))
            beat_symbols.append(sym)
            if sym in NORMAL_BEATS:
                annot_arr[s] = 1
                beat_type_arr[s] = 1
            elif sym in PVC_BEATS:
                annot_arr[s] = 2
                beat_type_arr[s] = 2
            elif sym in APB_BEATS:
                annot_arr[s] = 2
                beat_type_arr[s] = 3
            else:
                annot_arr[s] = 2
                beat_type_arr[s] = 4
    except Exception:
        pass

    rr_raw = np.full(T, np.nan, dtype=np.float32)
    drr_raw = np.full(T, np.nan, dtype=np.float32)
    amp_raw = np.full(T, np.nan, dtype=np.float32)
    damp_raw = np.full(T, np.nan, dtype=np.float32)
    # [is_beat, is_normal, is_abnormal, rr_norm, drr_norm, amp_norm, damp_norm]
    beat_event = np.zeros((T, 7), dtype=np.float32)

    if len(beat_pos) >= 2:
        bs = np.array(beat_pos, dtype=np.int64)
        rr = np.diff(bs) / float(SAMPLING_RATE)
        rr = np.concatenate([[rr[0]], rr]).astype(np.float32)
        drr = np.concatenate([[0.0], np.diff(rr)]).astype(np.float32)
        amp = signal[bs, 0].astype(np.float32)
        damp = np.concatenate([[0.0], np.diff(amp)]).astype(np.float32)

        rr_n = _zscore(rr)
        drr_n = _zscore(drr)
        amp_n = _zscore(amp)
        damp_n = _zscore(damp)

        rr_raw[bs] = rr_n
        drr_raw[bs] = drr_n
        amp_raw[bs] = amp_n
        damp_raw[bs] = damp_n

        beat_event[bs, 0] = 1.0
        beat_event[bs, 1] = (annot_arr[bs] == 1).astype(np.float32)
        beat_event[bs, 2] = (annot_arr[bs] == 2).astype(np.float32)
        beat_event[bs, 3] = rr_n
        beat_event[bs, 4] = drr_n
        beat_event[bs, 5] = amp_n
        beat_event[bs, 6] = damp_n
    elif len(beat_pos) == 1:
        bs = np.array(beat_pos, dtype=np.int64)
        amp = signal[bs, 0].astype(np.float32)
        amp_n = _zscore(amp)
        amp_raw[bs] = amp_n
        beat_event[bs, 0] = 1.0
        beat_event[bs, 1] = (annot_arr[bs] == 1).astype(np.float32)
        beat_event[bs, 2] = (annot_arr[bs] == 2).astype(np.float32)
        beat_event[bs, 5] = amp_n

    rr_ff, rr_mask = _ffill_with_mask(rr_raw)
    drr_ff, drr_mask = _ffill_with_mask(drr_raw)
    amp_ff, amp_mask = _ffill_with_mask(amp_raw)
    damp_ff, damp_mask = _ffill_with_mask(damp_raw)

    feat_dense = np.stack([rr_ff, drr_ff, amp_ff, damp_ff], axis=1)
    feat_obs_mask = np.stack([rr_mask, drr_mask, amp_mask, damp_mask], axis=1)

    abnormal_ratio = float(np.mean(annot_arr[annot_arr > 0] == 2)) if np.any(annot_arr > 0) else 0.0
    record_group = _classify_record_group(beat_symbols, abnormal_ratio)
    meta = {
        "abnormal_ratio": abnormal_ratio,
        "beat_count": float(len(beat_pos)),
    }

    return RecordData(
        name=name,
        signal=signal,
        annot_arr=annot_arr,
        beat_type_arr=beat_type_arr,
        feat_dense=feat_dense,
        feat_obs_mask=feat_obs_mask,
        beat_event=beat_event,
        record_group=record_group,
        meta=meta,
    )


class ECGAugmentedDataset(Dataset):
    def __init__(
        self,
        records: List[str],
        data_folder: str = DATA_FOLDER,
        input_len: int = INPUT_LEN,
        forecast_len: int = FORECAST_LEN,
        stride: int = STRIDE,
        augment: bool = False,
        seed: int = SEED,
    ) -> None:
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.augment = augment
        self.seed = seed

        self._records: List[RecordData] = []
        self._index: List[Tuple[int, int]] = []
        self.window_subtypes: List[int] = []
        self.record_groups: List[int] = []
        self.record_names: List[str] = []
        self.bucket_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for r_idx, name in enumerate(records):
            rec = _load_record(data_folder, name)
            self._records.append(rec)
            total_window = input_len + forecast_len
            for start in range(0, rec.signal.shape[0] - total_window + 1, stride):
                end = start + input_len
                forecast_end = end + forecast_len

                past_has_abn = bool(np.any(rec.annot_arr[start:end] == 2))
                future_has_abn = bool(np.any(rec.annot_arr[end:forecast_end] == 2))

                if (not past_has_abn) and (not future_has_abn):
                    subtype = WINDOW_SUBTYPES["normal"]
                elif (not past_has_abn) and future_has_abn:
                    subtype = WINDOW_SUBTYPES["pre_event"]
                elif past_has_abn and future_has_abn:
                    subtype = WINDOW_SUBTYPES["persistent_abnormal"]
                else:
                    subtype = WINDOW_SUBTYPES["recovery"]

                idx = len(self._index)
                self._index.append((r_idx, start))
                self.window_subtypes.append(subtype)
                self.record_groups.append(rec.record_group)
                self.record_names.append(rec.name)
                self.bucket_to_indices[(subtype, rec.record_group)].append(idx)

        self.sample_weights = self._build_balanced_weights()
        self.bucket_summary = {k: len(v) for k, v in self.bucket_to_indices.items()}
        self.normal_indices = [
            i for i, subtype in enumerate(self.window_subtypes) if subtype == WINDOW_SUBTYPES["normal"]
        ]

    def _build_balanced_weights(self) -> torch.DoubleTensor:
        n = len(self._index)
        if n == 0:
            return torch.ones(0, dtype=torch.double)

        bucket_counts = {k: len(v) for k, v in self.bucket_to_indices.items()}
        nonempty_buckets = max(len(bucket_counts), 1)
        weights = np.zeros(n, dtype=np.float64)

        # Balance directly on subtype x record-group buckets.
        for bucket, indices in self.bucket_to_indices.items():
            bucket_weight = n / (nonempty_buckets * max(len(indices), 1))
            weights[indices] = bucket_weight

        # Prevent a handful of rare windows from dominating the epoch.
        positive = weights[weights > 0]
        med = np.median(positive) if len(positive) else 1.0
        weights = np.clip(weights, med * 0.33, med * 3.0)
        return torch.from_numpy(weights).double()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        r_idx, start = self._index[idx]
        rec = self._records[r_idx]
        end = start + self.input_len
        forecast_end = end + self.forecast_len

        x_signal = rec.signal[start:end].copy()
        x_annot = rec.annot_arr[start:end].copy()
        x_feat = rec.feat_dense[start:end].copy()
        x_feat_mask = rec.feat_obs_mask[start:end].copy()
        x_beat_event = rec.beat_event[start:end].copy()
        y_signal = rec.signal[end:forecast_end, 0].copy()

        fc_types = rec.beat_type_arr[end:forecast_end]
        nonzero = fc_types[fc_types > 0]
        y_beat_type = int(np.bincount(nonzero, minlength=5)[1:].argmax() + 1) if len(nonzero) > 0 else 0
        y_risk = float(np.any(rec.annot_arr[end:forecast_end] == 2))

        beat_pos_win = np.where(rec.annot_arr[start:end] > 0)[0]
        if len(beat_pos_win) >= 2:
            rr = np.diff(beat_pos_win) / float(SAMPLING_RATE)
            sdnn = float(rr.std())
            rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
            pnn50 = float(np.mean(np.abs(np.diff(rr)) > 0.050)) if len(rr) > 1 else 0.0
        else:
            sdnn = rmssd = pnn50 = 0.0
        x_hrv = np.array([sdnn, rmssd, pnn50], dtype=np.float32)

        if self.augment:
            # stochastic per draw, so repeated sampling of the same window still yields variety
            rng = np.random.default_rng()
            x_signal = _simple_signal_augment(x_signal, rng)
            x_feat, x_beat_event = _augment_sparse_features(x_feat, x_feat_mask, x_beat_event, rng)

        return {
            "x_signal": torch.from_numpy(x_signal).float(),
            "x_annot": torch.from_numpy(x_annot).long(),
            "x_feat": torch.from_numpy(x_feat).float(),
            "x_feat_mask": torch.from_numpy(x_feat_mask).float(),
            "x_beat_event": torch.from_numpy(x_beat_event).float(),
            "y_signal": torch.from_numpy(y_signal).float(),
            "y_risk": torch.tensor(y_risk, dtype=torch.float32),
            "y_beat_type": torch.tensor(y_beat_type, dtype=torch.int64),
            "x_hrv": torch.from_numpy(x_hrv).float(),
            "window_subtype": torch.tensor(self.window_subtypes[idx], dtype=torch.int64),
            "record_group": torch.tensor(self.record_groups[idx], dtype=torch.int64),
            "record_name": rec.name,
        }

    def summary(self) -> str:
        subtype_counts = Counter(self.window_subtypes)
        group_counts = Counter(self.record_groups)
        lines = ["Dataset summary"]
        lines.append("Window subtypes:")
        for k in range(len(SUBTYPE_NAMES)):
            lines.append(f"  - {SUBTYPE_NAMES[k]}: {subtype_counts.get(k, 0)}")
        lines.append("Record groups:")
        for k in range(len(RECORD_GROUP_NAMES)):
            lines.append(f"  - {RECORD_GROUP_NAMES[k]}: {group_counts.get(k, 0)}")
        lines.append("Bucket counts (subtype x group):")
        for st in range(len(SUBTYPE_NAMES)):
            for rg in range(len(RECORD_GROUP_NAMES)):
                v = self.bucket_summary.get((st, rg), 0)
                if v > 0:
                    lines.append(f"  - {SUBTYPE_NAMES[st]} x {RECORD_GROUP_NAMES[rg]}: {v}")
        return "\n".join(lines)


def get_augmented_dataloaders(
    data_folder: str = DATA_FOLDER,
    input_len: int = INPUT_LEN,
    forecast_len: int = FORECAST_LEN,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    split: float = TRAIN_VAL_SPLIT,
    seed: int = SEED,
):
    rng = np.random.default_rng(seed)
    records = get_record_names(data_folder)
    records = list(rng.permutation(records))

    n_train = int(len(records) * split)
    train_records = records[:n_train]
    val_records = records[n_train:]

    train_ds = ECGAugmentedDataset(
        train_records,
        data_folder=data_folder,
        input_len=input_len,
        forecast_len=forecast_len,
        stride=stride,
        augment=True,
        seed=seed,
    )
    val_ds = ECGAugmentedDataset(
        val_records,
        data_folder=data_folder,
        input_len=input_len,
        forecast_len=forecast_len,
        stride=stride,
        augment=False,
        seed=seed,
    )

    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds, val_ds


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Subtype-aware augmented MIT-BIH dataset (patched).")
    parser.add_argument("--data-dir", type=str, default=DATA_FOLDER)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--forecast-len", type=int, default=FORECAST_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_loader, val_loader, train_ds, val_ds = get_augmented_dataloaders(
        data_folder=args.data_dir,
        input_len=args.input_len,
        forecast_len=args.forecast_len,
        stride=args.stride,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"Train windows: {len(train_ds):,}")
    print(train_ds.summary())
    print(f"Val windows: {len(val_ds):,}")
    print(val_ds.summary())

    batch = next(iter(train_loader))
    print("One batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    main()

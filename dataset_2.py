#!/usr/bin/env python3
"""
MIT-BIH Arrhythmia Database quality-control script.

What it does
------------
1) Scans every record listed in RECORDS under the MIT-BIH dataset directory.
2) Prints easy-to-interpret per-record signal-quality and annotation-quality stats.
3) Automatically picks:
   - one record with no abnormal beats (or the lowest abnormal burden), and
   - one record with abnormal beats / high arrhythmia burden,
   then saves ECG plots with beat annotations for both.

Usage
-----
python mitdb_qc.py --data-dir /path/to/mitdb/1.0.0 --out-dir qc_outputs

Expected MIT-BIH layout
-----------------------
<data-dir>/RECORDS
<data-dir>/100.dat, 100.hea, 100.atr, ...

Dependencies
------------
- wfdb
- matplotlib
- numpy (normally installed with wfdb)
"""

from __future__ import annotations

import argparse
import os
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import wfdb

# MIT-BIH abnormal-beat symbols commonly treated as arrhythmic/non-normal.
ABNORMAL_BEAT_SYMBOLS = {
    "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "x", "Q", "|", "!", "[", "]"
}

# These are normal or non-beat rhythm/comment symbols that should not be counted as beats.
NON_BEAT_SYMBOLS = {
    "+", "~", "|", "s", "T", "*", "D", "=", '"', "@", "^", "`", "'", "(", ")"
}

# Noise / quality / unreadable-style annotations we can track separately when present.
QUALITY_SYMBOLS = {"~", "U"}


def read_records_list(data_dir: str) -> List[str]:
    records_file = os.path.join(data_dir, "RECORDS")
    if not os.path.exists(records_file):
        raise FileNotFoundError(
            f"Could not find RECORDS in {data_dir}. Point --data-dir at the MIT-BIH folder containing RECORDS."
        )
    with open(records_file, "r", encoding="utf-8") as f:
        records = [line.strip() for line in f if line.strip()]
    if not records:
        raise RuntimeError("RECORDS exists but is empty.")
    return records


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def estimate_signal_quality(sig: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Simple, easy-to-interpret QC stats for one ECG channel.
    These are heuristics rather than clinical-quality signal-quality indices.
    """
    x = np.asarray(sig, dtype=float)
    n = len(x)
    if n == 0:
        return {}

    # Low-frequency baseline estimate over 0.6 s.
    baseline = moving_average(x, max(3, int(round(0.6 * fs))))
    hp = x - baseline

    # Derivative-based rough noise indicator.
    diff = np.diff(x, prepend=x[0])

    # Flatline-like fraction: very tiny changes over short scale.
    flatline_fraction = float(np.mean(np.abs(diff) < 1e-6))

    # Very large-slope fraction: likely motion/noise spikes.
    high_diff_threshold = np.percentile(np.abs(diff), 99.5)
    if high_diff_threshold == 0:
        spike_fraction = 0.0
    else:
        spike_fraction = float(np.mean(np.abs(diff) >= high_diff_threshold))

    # Baseline wander estimate (std of smoothed low-frequency trend).
    baseline_wander_std = float(np.std(baseline))

    return {
        "nan_fraction": float(np.mean(~np.isfinite(x))),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "range": float(np.nanmax(x) - np.nanmin(x)),
        "rms": float(np.sqrt(np.nanmean(np.square(x)))),
        "baseline_wander_std": baseline_wander_std,
        "hp_std": float(np.std(hp)),
        "flatline_fraction": flatline_fraction,
        "spike_fraction": spike_fraction,
    }


def classify_annotation_symbols(symbols: List[str]) -> Tuple[List[int], List[str], List[str]]:
    """Return indices/symbols for beat annotations and non-beat annotations."""
    beat_indices = []
    beat_symbols = []
    nonbeat_symbols = []

    for i, sym in enumerate(symbols):
        # For MIT-BIH, actual beats are mostly single-character symbols such as N, V, A, etc.
        # Rhythm-change markers and comments often appear in aux_note rather than in symbol.
        if sym in NON_BEAT_SYMBOLS:
            nonbeat_symbols.append(sym)
            continue
        # Keep common beat-like symbols as beats.
        beat_indices.append(i)
        beat_symbols.append(sym)

    return beat_indices, beat_symbols, nonbeat_symbols


def summarize_record(data_dir: str, record_name: str) -> Dict[str, object]:
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    sig = record.p_signal
    fs = int(record.fs)
    n_samples, n_channels = sig.shape
    duration_sec = n_samples / float(fs)
    sig_names = list(record.sig_name)

    channel_stats = []
    for ch in range(n_channels):
        channel_stats.append(estimate_signal_quality(sig[:, ch], fs))

    beat_idx, beat_symbols, nonbeat_symbols = classify_annotation_symbols(list(ann.symbol))
    beat_samples = np.array([ann.sample[i] for i in beat_idx], dtype=int)
    beat_symbols_arr = np.array(beat_symbols, dtype=object)

    beat_counter = Counter(beat_symbols)
    abnormal_mask = np.array([sym in ABNORMAL_BEAT_SYMBOLS for sym in beat_symbols_arr], dtype=bool)
    abnormal_count = int(np.sum(abnormal_mask))
    beat_count = int(len(beat_samples))
    normal_count = int(beat_counter.get("N", 0) + beat_counter.get(".", 0))

    rr_intervals_sec = np.diff(beat_samples) / float(fs) if beat_count >= 2 else np.array([], dtype=float)
    rr_stats = {
        "median_rr_sec": float(np.median(rr_intervals_sec)) if rr_intervals_sec.size else math.nan,
        "mean_rr_sec": float(np.mean(rr_intervals_sec)) if rr_intervals_sec.size else math.nan,
        "std_rr_sec": float(np.std(rr_intervals_sec)) if rr_intervals_sec.size else math.nan,
        "min_rr_sec": float(np.min(rr_intervals_sec)) if rr_intervals_sec.size else math.nan,
        "max_rr_sec": float(np.max(rr_intervals_sec)) if rr_intervals_sec.size else math.nan,
        "short_rr_lt_0p3s": int(np.sum(rr_intervals_sec < 0.30)) if rr_intervals_sec.size else 0,
        "long_rr_gt_2s": int(np.sum(rr_intervals_sec > 2.0)) if rr_intervals_sec.size else 0,
    }

    bpm = 60.0 / rr_stats["median_rr_sec"] if rr_intervals_sec.size and rr_stats["median_rr_sec"] > 0 else math.nan

    # Annotation quality heuristics.
    out_of_bounds_beats = int(np.sum((beat_samples < 0) | (beat_samples >= n_samples)))
    duplicate_beats = int(np.sum(np.diff(beat_samples) == 0)) if beat_count >= 2 else 0
    decreasing_samples = int(np.sum(np.diff(beat_samples) < 0)) if beat_count >= 2 else 0
    quality_ann_count = sum(sym in QUALITY_SYMBOLS for sym in ann.symbol)
    unreadable_aux = sum("unreadable" in (note or "").lower() for note in ann.aux_note)
    noise_aux = sum("noise" in (note or "").lower() for note in ann.aux_note)

    # Rhythm labels are often encoded in aux_note on '+' annotation rows.
    rhythm_notes = [note.strip() for note in ann.aux_note if note and note.strip().startswith("(")]
    rhythm_counter = Counter(rhythm_notes)

    return {
        "record": record_name,
        "fs": fs,
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "sig_names": sig_names,
        "channel_stats": channel_stats,
        "beat_count": beat_count,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "abnormal_fraction": (abnormal_count / beat_count) if beat_count else math.nan,
        "beat_counter": beat_counter,
        "rr_stats": rr_stats,
        "median_bpm": bpm,
        "quality_ann_count": quality_ann_count,
        "unreadable_aux": unreadable_aux,
        "noise_aux": noise_aux,
        "out_of_bounds_beats": out_of_bounds_beats,
        "duplicate_beats": duplicate_beats,
        "decreasing_samples": decreasing_samples,
        "rhythm_counter": rhythm_counter,
        "ann": ann,
        "record_obj": record,
    }


def print_record_summary(summary: Dict[str, object]) -> None:
    print("=" * 88)
    print(f"Record {summary['record']}")
    print(f"  Duration: {summary['duration_sec'] / 60:.2f} min | fs: {summary['fs']} Hz | channels: {summary['n_channels']} | leads: {', '.join(summary['sig_names'])}")

    for i, ch_stats in enumerate(summary["channel_stats"]):
        lead = summary["sig_names"][i]
        print(
            f"  Lead {i+1} ({lead}): mean={ch_stats['mean']:.4f}, std={ch_stats['std']:.4f}, "
            f"range={ch_stats['range']:.4f}, baseline_wander_std={ch_stats['baseline_wander_std']:.4f}, "
            f"flatline_frac={100*ch_stats['flatline_fraction']:.3f}%, spike_frac={100*ch_stats['spike_fraction']:.3f}%"
        )

    rr = summary["rr_stats"]
    print(
        f"  Beats: total={summary['beat_count']} | normal={summary['normal_count']} | abnormal={summary['abnormal_count']} "
        f"({100*summary['abnormal_fraction']:.2f}%) | median BPM={summary['median_bpm']:.1f}"
    )
    print(
        f"  RR intervals: median={rr['median_rr_sec']:.3f}s, mean={rr['mean_rr_sec']:.3f}s, std={rr['std_rr_sec']:.3f}s, "
        f"min={rr['min_rr_sec']:.3f}s, max={rr['max_rr_sec']:.3f}s, short<0.3s={rr['short_rr_lt_0p3s']}, long>2s={rr['long_rr_gt_2s']}"
    )

    common_beats = summary["beat_counter"].most_common(8)
    common_beats_str = ", ".join(f"{sym}:{cnt}" for sym, cnt in common_beats) if common_beats else "None"
    print(f"  Beat label mix (top): {common_beats_str}")

    rhythm_top = summary["rhythm_counter"].most_common(5)
    rhythm_str = ", ".join(f"{lab}:{cnt}" for lab, cnt in rhythm_top) if rhythm_top else "None found"
    print(f"  Rhythm annotations (top): {rhythm_str}")

    print(
        f"  Annotation QC flags: out_of_bounds={summary['out_of_bounds_beats']}, duplicate_samples={summary['duplicate_beats']}, "
        f"decreasing_samples={summary['decreasing_samples']}, quality_symbols={summary['quality_ann_count']}, "
        f"noise_notes={summary['noise_aux']}, unreadable_notes={summary['unreadable_aux']}"
    )


def choose_example_records(summaries: List[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    # Mostly normal: prefer zero abnormal beats, else smallest abnormal fraction.
    no_arr = sorted(summaries, key=lambda s: (s["abnormal_count"] != 0, s["abnormal_fraction"], s["record"]))[0]
    # Arrhythmic: prefer highest abnormal fraction, then highest abnormal count.
    arr = sorted(summaries, key=lambda s: (s["abnormal_fraction"], s["abnormal_count"]), reverse=True)[0]
    return no_arr, arr


def find_interesting_window(summary: Dict[str, object], window_sec: float = 10.0) -> Tuple[int, int]:
    ann = summary["ann"]
    record = summary["record_obj"]
    fs = int(record.fs)
    n_samples = record.p_signal.shape[0]

    beat_idx, beat_symbols, _ = classify_annotation_symbols(list(ann.symbol))
    beat_samples = np.array([ann.sample[i] for i in beat_idx], dtype=int)
    beat_symbols_arr = np.array(beat_symbols, dtype=object)

    abnormal_positions = beat_samples[np.array([sym in ABNORMAL_BEAT_SYMBOLS for sym in beat_symbols_arr], dtype=bool)]
    if abnormal_positions.size > 0:
        center = int(abnormal_positions[len(abnormal_positions) // 2])
    elif beat_samples.size > 0:
        center = int(beat_samples[len(beat_samples) // 2])
    else:
        center = n_samples // 2

    half = int(round(window_sec * fs / 2))
    start = max(0, center - half)
    end = min(n_samples, center + half)
    return start, end


def plot_record_with_annotations(summary: Dict[str, object], out_path: str, window_sec: float = 10.0) -> None:
    record = summary["record_obj"]
    ann = summary["ann"]
    fs = int(record.fs)
    start, end = find_interesting_window(summary, window_sec=window_sec)

    sig = record.p_signal[start:end, :]
    t = np.arange(start, end) / float(fs)
    rel_t = t - t[0]

    ann_mask = (ann.sample >= start) & (ann.sample < end)
    ann_samples = ann.sample[ann_mask]
    ann_symbols = np.array(ann.symbol, dtype=object)[ann_mask]

    n_channels = sig.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    y_offsets = []
    for ch in range(n_channels):
        ax = axes[ch]
        y = sig[:, ch]
        ax.plot(rel_t, y, linewidth=1.0)
        ax.set_ylabel(f"{record.sig_name[ch]}\n(mV)")
        ax.grid(True, alpha=0.25)
        y_offsets.append(np.nanmax(y) if np.isfinite(np.nanmax(y)) else 0.0)

        for s, sym in zip(ann_samples, ann_symbols):
            x = (s / float(fs)) - t[0]
            color = "red" if sym in ABNORMAL_BEAT_SYMBOLS else "green"
            ax.axvline(x=x, linestyle="--", linewidth=0.8, alpha=0.5, color=color)
            ax.text(x, y_offsets[-1], sym, fontsize=8, rotation=90, va="bottom", ha="center", color=color)

    fig.suptitle(
        f"Record {summary['record']} | abnormal beats={summary['abnormal_count']} ({100*summary['abnormal_fraction']:.2f}%)\n"
        f"Window: {window_sec:.0f}s with beat annotations (green=normal/other, red=abnormal)"
    )
    axes[-1].set_xlabel("Time within window (s)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG quality-control script")
    parser.add_argument("--data-dir", required=True, help="Path to MIT-BIH directory containing RECORDS")
    parser.add_argument("--out-dir", default="qc_outputs", help="Directory to save output plots")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Plot window length in seconds")
    args = parser.parse_args()

    records = read_records_list(args.data_dir)
    print(f"Found {len(records)} records in {args.data_dir}\n")

    summaries: List[Dict[str, object]] = []
    for rec in records:
        try:
            summary = summarize_record(args.data_dir, rec)
            summaries.append(summary)
            print_record_summary(summary)
        except Exception as e:
            print("=" * 88)
            print(f"Record {rec}")
            print(f"  ERROR while reading record: {e}")

    if not summaries:
        raise RuntimeError("No records were successfully read.")

    normal_rec, arr_rec = choose_example_records(summaries)
    os.makedirs(args.out_dir, exist_ok=True)

    normal_plot = os.path.join(args.out_dir, f"record_{normal_rec['record']}_mostly_normal.png")
    arr_plot = os.path.join(args.out_dir, f"record_{arr_rec['record']}_arrhythmia.png")

    plot_record_with_annotations(normal_rec, normal_plot, window_sec=args.window_sec)
    plot_record_with_annotations(arr_rec, arr_plot, window_sec=args.window_sec)

    print("\n" + "#" * 88)
    print("Example plots saved")
    print(f"  Mostly normal example : {normal_plot} (record {normal_rec['record']})")
    print(f"  Arrhythmia example    : {arr_plot} (record {arr_rec['record']})")
    print("#" * 88)


if __name__ == "__main__":
    main()

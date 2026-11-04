"""Build the pretraining cache from the TUH EEG Corpus (v2.0.1, EDF files).

  python preprocess_tuh.py --tuh_root /data/tuh_eeg/v2.0.1/edf --cache_dir cache/tuh_200hz [--workers 16]

Per recording: the 19 referential 10-20 channels (pretraining order, see CHANNEL_MAP); recordings of 130 s or less, or
missing a channel, are skipped; the first and last 60 s of the others are trimmed; 60 Hz IIR notch (Q 30) and 5th-order Butterworth
0.1-75 Hz band-pass (both zero phase) at the native rate; polyphase resampling to 200 Hz; non-overlapping 6 s epochs;
epochs with |x| > 100 uV (or NaN/inf) are rejected; scaling x1e4 (volts -> uV/100). Each recording is written to
<cache_dir>/<params hash>/<edf name>_<path hash>.npz with array 'epochs' [N, 19, 1200] float32, next to
preprocess_params.json.
"""
import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from math import gcd

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

mne.set_log_level('ERROR')

# PARAMS is written to preprocess_params.json and names the cache directory (its hash); the duration rule actually applied
# is the one in recording_epochs (keep recordings longer than 2 x trim_sec + 10 s).
PARAMS = {"version": 2, "target_sfreq": 200, "epoch_sec": 6.0, "bandpass": [0.1, 75.0], "amp_reject": 100e-6,
          "notch_freq": 60.0, "trim_sec": 60.0, "normalize_scale": 1e4, "min_duration_sec": 300}
CHANNEL_MAP = {
    'FP1': ['EEG FP1-REF', 'EEG FP1-LE', 'FP1'], 'FP2': ['EEG FP2-REF', 'EEG FP2-LE', 'FP2'],
    'F7': ['EEG F7-REF', 'EEG F7-LE', 'F7'], 'F3': ['EEG F3-REF', 'EEG F3-LE', 'F3'],
    'FZ': ['EEG FZ-REF', 'EEG FZ-LE', 'FZ'], 'F4': ['EEG F4-REF', 'EEG F4-LE', 'F4'],
    'F8': ['EEG F8-REF', 'EEG F8-LE', 'F8'], 'T3': ['EEG T3-REF', 'EEG T3-LE', 'T3', 'EEG T7-REF', 'EEG T7-LE'],
    'C3': ['EEG C3-REF', 'EEG C3-LE', 'C3'], 'CZ': ['EEG CZ-REF', 'EEG CZ-LE', 'CZ'],
    'C4': ['EEG C4-REF', 'EEG C4-LE', 'C4'], 'T4': ['EEG T4-REF', 'EEG T4-LE', 'T4', 'EEG T8-REF', 'EEG T8-LE'],
    'P3': ['EEG P3-REF', 'EEG P3-LE', 'P3'], 'PZ': ['EEG PZ-REF', 'EEG PZ-LE', 'PZ'],
    'P4': ['EEG P4-REF', 'EEG P4-LE', 'P4'], 'T5': ['EEG T5-REF', 'EEG T5-LE', 'T5', 'EEG P7-REF', 'EEG P7-LE'],
    'T6': ['EEG T6-REF', 'EEG T6-LE', 'T6', 'EEG P8-REF', 'EEG P8-LE'], 'O1': ['EEG O1-REF', 'EEG O1-LE', 'O1'],
    'O2': ['EEG O2-REF', 'EEG O2-LE', 'O2'],
}


def recording_epochs(edf_path, P=PARAMS):
    """One EDF -> float32 [N, 19, 1200] (possibly N = 0), or None if the recording is skipped."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    pick = []
    for std in CHANNEL_MAP:
        hit = next((a for a in CHANNEL_MAP[std] if a in raw.ch_names), None)
        if hit is None:
            return None
        pick.append(hit)
    raw.pick_channels(pick, ordered=True)
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    trim = int(P["trim_sec"] * sfreq)
    if data.shape[1] <= 2 * trim + int(sfreq * 10):
        return None
    data = data[:, trim:-trim]
    b, a = iirnotch(P["notch_freq"], Q=30.0, fs=sfreq)
    data = filtfilt(b, a, data, axis=-1).astype(np.float32)
    nyq = sfreq / 2.0
    sos = butter(5, [max(P["bandpass"][0] / nyq, 0.001), min(P["bandpass"][1] / nyq, 0.999)], btype='band', output='sos')
    data = sosfiltfilt(sos, data, axis=-1).astype(np.float32)
    if sfreq != P["target_sfreq"]:
        g = gcd(int(P["target_sfreq"]), int(sfreq))
        data = resample_poly(data, int(P["target_sfreq"]) // g, int(sfreq) // g, axis=-1).astype(np.float32)
    n = int(P["epoch_sec"] * P["target_sfreq"])
    epochs = [data[:, i * n:(i + 1) * n] for i in range(data.shape[1] // n)]
    keep = [e * P["normalize_scale"] for e in epochs
            if np.abs(e).max() <= P["amp_reject"] and np.isfinite(e).all()]
    return np.stack(keep).astype(np.float32) if keep else np.zeros((0, 19, n), np.float32)


def _one(args):
    edf_path, out_dir = args
    name = os.path.splitext(os.path.basename(edf_path))[0]
    out = os.path.join(out_dir, f"{name}_{hashlib.md5(edf_path.encode()).hexdigest()[:16]}.npz")
    if os.path.exists(out):
        return "cached"
    try:
        ep = recording_epochs(edf_path)
    except Exception:
        return "error"
    if ep is None or len(ep) == 0:
        return "skipped"
    np.savez(out + ".tmp.npz", epochs=ep)
    os.replace(out + ".tmp.npz", out)
    return "written"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuh_root", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out_dir = os.path.join(a.cache_dir, hashlib.md5(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:12])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preprocess_params.json"), "w") as fh:
        json.dump(PARAMS, fh, indent=2)
    files = sorted(glob.glob(os.path.join(a.tuh_root, "**", "*.edf"), recursive=True))
    print(f"{len(files):,} EDF files", flush=True)
    counts = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_one, [(f, out_dir) for f in files], chunksize=8)):
            counts[r] = counts.get(r, 0) + 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{len(files):,} {counts}", flush=True)
    print(f"done: {counts} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

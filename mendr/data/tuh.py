"""TUAB (abnormal EEG) and TUEV (EEG events) from the raw TUH EDF files, 19 referential channels at 200 Hz.

Splits: the official eval set is the test set; a validation set of subjects is carved from the official train set
(subject lists in mendr/data/splits/{tuab,tuev}.json, so every model sees the same partition).

TUAB: per recording, 60 Hz notch and 0.1-75 Hz band-pass (MNE defaults), resampling to 200 Hz, non-overlapping 6 s epochs,
scaling x1e4 (volts -> uV/100); epochs with NaN/inf, std < 1e-6 or |x| > 5 are dropped. Label = recording label.
TUEV: per recording, 60 Hz IIR notch (Q 30), 5th-order Butterworth 0.1-75 Hz (zero phase), polyphase resampling to 200 Hz;
one 5 s window centred on each annotated event (.rec file, classes 1-6 -> 0-5), scaled x1e4.
"""
import glob
import json
import os
import re
from collections import Counter
from math import gcd

import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from torch.utils.data import Dataset

mne.set_log_level('ERROR')

SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")

CHANNEL_MAP = {      # 19 channels in pretraining order: standard name -> accepted EDF names
    'FP1': ['EEG FP1-REF', 'EEG FP1-LE', 'FP1'],
    'FP2': ['EEG FP2-REF', 'EEG FP2-LE', 'FP2'],
    'F7': ['EEG F7-REF', 'EEG F7-LE', 'F7'],
    'F3': ['EEG F3-REF', 'EEG F3-LE', 'F3'],
    'FZ': ['EEG FZ-REF', 'EEG FZ-LE', 'FZ'],
    'F4': ['EEG F4-REF', 'EEG F4-LE', 'F4'],
    'F8': ['EEG F8-REF', 'EEG F8-LE', 'F8'],
    'T3': ['EEG T3-REF', 'EEG T3-LE', 'T3', 'EEG T7-REF', 'EEG T7-LE', 'T7'],
    'C3': ['EEG C3-REF', 'EEG C3-LE', 'C3'],
    'CZ': ['EEG CZ-REF', 'EEG CZ-LE', 'CZ'],
    'C4': ['EEG C4-REF', 'EEG C4-LE', 'C4'],
    'T4': ['EEG T4-REF', 'EEG T4-LE', 'T4', 'EEG T8-REF', 'EEG T8-LE', 'T8'],
    'P3': ['EEG P3-REF', 'EEG P3-LE', 'P3'],
    'PZ': ['EEG PZ-REF', 'EEG PZ-LE', 'PZ'],
    'P4': ['EEG P4-REF', 'EEG P4-LE', 'P4'],
    'T5': ['EEG T5-REF', 'EEG T5-LE', 'T5', 'EEG P7-REF', 'EEG P7-LE', 'P7'],
    'T6': ['EEG T6-REF', 'EEG T6-LE', 'T6', 'EEG P8-REF', 'EEG P8-LE', 'P8'],
    'O1': ['EEG O1-REF', 'EEG O1-LE', 'O1'],
    'O2': ['EEG O2-REF', 'EEG O2-LE', 'O2'],
}
CHANNELS_19 = list(CHANNEL_MAP)
_TUEV_ALIASES = {k: [a for a in v if a not in ('T7', 'T8', 'P7', 'P8')] for k, v in CHANNEL_MAP.items()}
_SUBJ_RE = re.compile(r"([a-z0-9]+)_s\d+_t\d+")


def load_split(task):
    with open(os.path.join(SPLIT_DIR, f"{task}.json")) as fh:
        return json.load(fh)


def subject_of(path):
    m = _SUBJ_RE.match(os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


# ── TUAB ─────────────────────────────────────────────────────────────────────
def tuab_epochs(edf_path, target_sfreq=200, epoch_sec=6.0, bandpass=(0.1, 75.0)):
    """One TUAB recording -> float32 [n_epochs, 19, 1200] or None."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    pick = []
    for ch in CHANNELS_19:
        mapped = next((a for a in CHANNEL_MAP[ch] if a in raw.ch_names), None)
        if mapped is None:
            return None
        pick.append(mapped)
    try:
        raw.pick_channels(pick, ordered=True)
    except ValueError:
        return None
    try:
        raw.notch_filter(60.0, verbose=False)
    except Exception:
        pass
    try:
        raw.filter(bandpass[0], bandpass[1], verbose=False)
    except Exception:
        return None
    if raw.info['sfreq'] != target_sfreq:
        try:
            raw.resample(target_sfreq, verbose=False)
        except Exception:
            return None
    data = raw.get_data().astype(np.float32)
    epoch_samples = int(epoch_sec * target_sfreq)
    n_epochs = data.shape[1] // epoch_samples
    if n_epochs == 0:
        return None
    epochs = data[:, :n_epochs * epoch_samples].reshape(19, n_epochs, epoch_samples).transpose(1, 0, 2)
    epochs = epochs * 1e4
    good = [i for i in range(n_epochs)
            if not (np.isnan(epochs[i]).any() or np.isinf(epochs[i]).any())
            and not epochs[i].std() < 1e-6 and not np.abs(epochs[i]).max() > 5.0]
    return epochs[good] if good else None


def _tuab_files(root, split):
    out = []
    for label_name, label_id in (("normal", 0), ("abnormal", 1)):
        d = os.path.join(root, split, label_name, "01_tcp_ar")
        if not os.path.isdir(d):
            d = os.path.join(root, split, label_name)
        if not os.path.isdir(d):
            continue
        out += [(os.path.join(d, f), label_id) for f in sorted(os.listdir(d)) if f.endswith(".edf")]
    return out


def load_tuab(root, subjects, split, limit_files=None, verbose=True):
    subjects = set(subjects)
    files = [(p, y) for p, y in _tuab_files(root, split) if subject_of(p) in subjects]
    if limit_files:
        pos = [f for f in files if f[1] == 1][:max(1, limit_files // 2)]
        neg = [f for f in files if f[1] == 0][:max(1, limit_files - len(pos))]
        files = neg + pos
    X, Y, S = [], [], []
    for p, y in files:
        ep = tuab_epochs(p)
        if ep is None or len(ep) == 0:
            continue
        X.append(ep); Y += [y] * len(ep); S += [subject_of(p)] * len(ep)
    if verbose:
        print(f"  [tuab/{split}] {len(files)} EDF files from {len(subjects)} subjects -> {sum(map(len, X)):,} epochs, "
              f"labels={sorted(Counter(Y).items())}", flush=True)
    if not X:
        return torch.zeros(0, 19, 1200), torch.zeros(0, dtype=torch.long), []
    return torch.from_numpy(np.concatenate(X, axis=0)), torch.tensor(Y, dtype=torch.long), S


# ── TUEV ─────────────────────────────────────────────────────────────────────
def tuev_windows(edf_path, rec_path, target_sfreq=200, bandpass=(0.1, 75.0), epoch_sec=5.0):
    """One TUEV recording -> list of (float32 tensor [19, 1000], label 0-5)."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    pick = []
    for ch in CHANNELS_19:
        mapped = next((a for a in _TUEV_ALIASES[ch] if a in raw.ch_names), None)
        if mapped is None:
            return []
        pick.append(mapped)
    raw.pick_channels(pick, ordered=True)
    orig_sfreq = raw.info['sfreq']
    raw_data = raw.get_data()
    b, a = iirnotch(60.0, Q=30.0, fs=orig_sfreq)
    raw_data = filtfilt(b, a, raw_data, axis=-1).astype(np.float32)
    nyq = orig_sfreq / 2.0
    sos = butter(5, [max(bandpass[0] / nyq, 0.001), min(bandpass[1] / nyq, 0.999)], btype='band', output='sos')
    data = sosfiltfilt(sos, raw_data, axis=-1).astype(np.float32)
    if orig_sfreq != target_sfreq:
        g = gcd(int(target_sfreq), int(orig_sfreq))
        data = resample_poly(data, int(target_sfreq) // g, int(orig_sfreq) // g, axis=-1).astype(np.float32)
    try:
        events = np.genfromtxt(rec_path, delimiter=',')
    except Exception:
        return []
    if events.ndim == 1:
        events = events.reshape(1, -1)
    if events.shape[1] < 4:
        return []
    total, n = data.shape[1], int(epoch_sec * target_sfreq)
    out = []
    for i in range(events.shape[0]):
        label = int(events[i, 3])
        if label < 1 or label > 6:
            continue
        start = int((events[i, 1] + events[i, 2]) / 2.0 * target_sfreq) - n // 2
        if start < 0 or start + n > total:
            continue
        epoch = data[:, start:start + n]
        if epoch.shape[1] != n:
            continue
        epoch = epoch * 1e4
        if np.isnan(epoch).any() or np.isinf(epoch).any():
            continue
        out.append((torch.from_numpy(epoch), label - 1))
    return out


def _tuev_subject(root, edf_path, split):
    parts = os.path.relpath(edf_path, os.path.join(root, split)).split(os.sep)
    return parts[0] if len(parts) > 1 else subject_of(edf_path)


def load_tuev(root, subjects, split, limit_files=None, verbose=True):
    subjects = set(subjects)
    files = [p for p in sorted(glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True))
             if _tuev_subject(root, p, split) in subjects]
    if limit_files:
        files = files[:limit_files]
    X, Y, S = [], [], []
    for edf in files:
        rec = edf[:-4] + ".rec"
        if not os.path.exists(rec):
            continue
        try:
            samples = tuev_windows(edf, rec)
        except Exception:
            continue
        for x, label in samples:
            X.append(x); Y.append(label); S.append(_tuev_subject(root, edf, split))
    if verbose:
        print(f"  [tuev/{split}] {len(files)} EDF files from {len(subjects)} subjects -> {len(X):,} windows, "
              f"labels={sorted(Counter(Y).items())}", flush=True)
    if not X:
        return torch.zeros(0, 19, 1000), torch.zeros(0, dtype=torch.long), []
    return torch.stack(X), torch.tensor(Y, dtype=torch.long), S


# ── datasets ─────────────────────────────────────────────────────────────────
class ArrayDataset(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], int(self.labels[i])


def build_tuh_task(task, root, cache_dir=None, limit_files=None, verbose=True):
    """(train, val, test, n_classes, in_channels) for task 'tuab' or 'tuev'. `root` is the dataset's edf/ directory
    (containing train/ and eval/). Parsed splits are cached in `cache_dir` when given."""
    sp = load_split(task)
    loader = {"tuab": load_tuab, "tuev": load_tuev}[task]
    out = []
    for name, subjects, pool in (("train", sp["train_subjects"], "train"), ("val", sp["val_subjects"], "train"),
                                 ("test", sp["test_subjects"], "eval")):
        path = os.path.join(cache_dir, f"{task}_{name}{'_lim%d' % limit_files if limit_files else ''}.pt") if cache_dir else None
        if path and os.path.exists(path):
            obj = torch.load(path, map_location="cpu", mmap=True)      # memory-mapped: TUAB train is ~49 GB
            X, Y = obj["X"], obj["Y"]
            if verbose:
                print(f"  [{task}/{name}] {len(X):,} epochs from cache {path}", flush=True)
        else:
            X, Y, _ = loader(root, subjects, pool, limit_files=limit_files, verbose=verbose)
            if len(X) == 0:
                raise RuntimeError(f"{task}/{name}: no epochs loaded from {root} (wrong --data_root or unreadable files)")
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"X": X, "Y": Y}, path + ".tmp"); os.replace(path + ".tmp", path)
        out.append(ArrayDataset(X, Y))
    return (*out, {"tuab": 2, "tuev": 6}[task], 19)

"""ISRUC-Sleep subgroup 1: 5-class sleep staging, 6 channels (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1) at 200 Hz.

Per recording: 50 Hz IIR notch (Q 30), 5th-order Butterworth 0.3-35 Hz (zero phase), scaling x1e4; 30 s epochs with the
first annotator's labels (W, N1, N2, N3, REM -> 0-4; other labels dropped). Subjects 1-80 train, 81-90 validation, 91-100
test. Recordings whose files or channels are missing are skipped and reported.
"""
import os
import tempfile
import time
from collections import Counter

import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

mne.set_log_level('ERROR')

CHANNELS_A = ['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1']
CHANNELS_M = ['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']   # the same montage under mastoid names
SFREQ, EPOCH_SAMPLES = 200, 200 * 30
NUM_CLASSES = 5
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
SPLITS = {"train": list(range(1, 81)), "val": list(range(81, 91)), "test": list(range(91, 101))}


def _read_rec(path):
    """ISRUC .rec files are EDF; MNE reads them through a temporary .edf symlink."""
    tmp = os.path.join(tempfile.gettempdir(), f"isruc_{os.getpid()}_{os.path.basename(path)}.edf")
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        os.symlink(os.path.abspath(path), tmp)
        return mne.io.read_raw_edf(tmp, preload=True, verbose=False)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ISRUC(Dataset):
    """`root` contains {subject}.rec and {subject}_1.txt for subjects 1-100. metadata[i] = subject of sample i."""

    def __init__(self, root, split, cache_dir=None, bandpass=(0.3, 35.0), notch_freq=50.0, scale=1e4):
        subjects = SPLITS[split]
        self.samples, self.metadata, skipped = [], [], []
        cache_path = os.path.join(cache_dir, f"isruc_{split}.pt") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            obj = torch.load(cache_path, map_location="cpu")
            self.samples = [(obj["x"][i], int(obj["y"][i])) for i in range(len(obj["y"]))]
            self.metadata, skipped = list(obj["metadata"]), list(obj["skipped"])
            print(f"  [ISRUC {split}] {len(self.samples):,} epochs from cache {cache_path}", flush=True)
        else:
            nyq = SFREQ / 2.0
            sos = butter(5, [max(bandpass[0] / nyq, 0.001), min(bandpass[1] / nyq, 0.999)], btype='band', output='sos')
            b_notch, a_notch = iirnotch(notch_freq, Q=30, fs=SFREQ)
            for subj in subjects:
                rec_path, label_path = os.path.join(root, f'{subj}.rec'), os.path.join(root, f'{subj}_1.txt')
                if not os.path.exists(rec_path) or not os.path.exists(label_path):
                    skipped.append(subj)
                    continue
                labels = np.loadtxt(label_path, dtype=int)
                raw = None
                for attempt in range(5):                        # transient file-system errors: retry, then fail loudly
                    try:
                        raw = _read_rec(rec_path)
                        break
                    except Exception as e:
                        print(f"  [ISRUC] subject {subj}: cannot read .rec (attempt {attempt + 1}/5): {e}", flush=True)
                        time.sleep(30)
                if raw is None:
                    raise RuntimeError(f"[ISRUC] subject {subj}: unreadable ({rec_path})")
                assert int(raw.info['sfreq']) == SFREQ, f"subject {subj}: sfreq {raw.info['sfreq']}"
                if all(ch in raw.ch_names for ch in CHANNELS_A):
                    pick = CHANNELS_A
                elif all(ch in raw.ch_names for ch in CHANNELS_M):
                    pick = CHANNELS_M
                else:
                    print(f"  [ISRUC] skipping subject {subj}: channels missing", flush=True)
                    skipped.append(subj)
                    continue
                raw.pick_channels(pick, ordered=True)
                data = raw.get_data().astype(np.float64)
                data = filtfilt(b_notch, a_notch, data, axis=-1)
                data = sosfiltfilt(sos, data, axis=-1)
                data = data * scale
                n_epochs = min(len(labels), data.shape[1] // EPOCH_SAMPLES)
                for ep in range(n_epochs):
                    if labels[ep] not in LABEL_MAP:
                        continue
                    epoch = data[:, ep * EPOCH_SAMPLES:(ep + 1) * EPOCH_SAMPLES].astype(np.float32)
                    if epoch.shape[1] != EPOCH_SAMPLES:
                        continue
                    self.samples.append((torch.from_numpy(epoch), LABEL_MAP[labels[ep]]))
                    self.metadata.append(subj)
            if cache_path and self.samples:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"x": torch.stack([s[0] for s in self.samples]),
                            "y": torch.tensor([s[1] for s in self.samples], dtype=torch.long),
                            "metadata": self.metadata, "skipped": skipped}, cache_path + ".tmp")
                os.replace(cache_path + ".tmp", cache_path)
        print(f"  [ISRUC {split}] {len(self.samples):,} epochs from {len(subjects) - len(skipped)}/{len(subjects)} subjects, "
              f"labels={sorted(Counter(s[1] for s in self.samples).items())}" + (f", skipped {skipped}" if skipped else ""), flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

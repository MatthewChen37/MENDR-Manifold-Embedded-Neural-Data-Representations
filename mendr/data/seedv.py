"""SEED-V: 5-class emotion recognition, 62 channels, 1 s windows at 200 Hz.

Per .cnt recording (1000 Hz): 50 Hz IIR notch (Q 30), 5th-order Butterworth 0.1-75 Hz (zero phase), polyphase resampling to
200 Hz; non-overlapping 1 s windows inside each trial, scaled x1e4. Labels: happy 0, sad 1, neutral 2, disgust 3, fear 4.
Cross-trial split within every session of the 16 subjects (47 sessions; 7_1_20180411.cnt is not used): trials 0-4 train,
5-9 validation, 10-14 test.
"""
import os
from collections import Counter
from math import gcd

import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from torch.utils.data import Dataset

mne.set_log_level('ERROR')

CHANNELS_62 = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2',
]
SESSION_START_TIMES = {
    1: [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    2: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    3: [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
}
SESSION_END_TIMES = {
    1: [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    2: [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    3: [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066],
}
_H, _S, _N, _D, _F = 0, 1, 2, 3, 4
TRIAL_LABELS = {
    1: [_H, _F, _N, _S, _D] * 3,
    2: [_S, _F, _N, _D, _H, _H, _D, _N, _S, _F, _N, _H, _F, _S, _D],
    3: [_S, _F, _N, _D, _H, _H, _D, _N, _S, _F, _N, _H, _F, _S, _D],
}
SUBJECTS = list(range(1, 17))
EXCLUDED = {"7_1_20180411.cnt"}   # cannot be read as a Neuroscan .cnt file; not used in the paper's experiments
SPLIT_TRIALS = {"train": list(range(0, 5)), "val": list(range(5, 10)), "test": list(range(10, 15))}


def _load_cnt(path):
    try:
        return mne.io.read_raw_cnt(path, preload=True, verbose=False)
    except Exception:
        pass
    return mne.io.read_raw_ant(path, preload=True, verbose=False)


class SEEDV(Dataset):
    """`root` contains the raw {subject}_{session}_*.cnt files. metadata[i] = (subject, session, trial, window)."""

    def __init__(self, root, split, cache_dir=None, sfreq_target=200, window_sec=1.0, bandpass=(0.1, 75.0),
                 notch_freq=50.0, scale=1e4):
        trials = SPLIT_TRIALS[split]
        self.window_samples = int(sfreq_target * window_sec)
        self.samples, self.metadata = [], []
        cache_path = os.path.join(cache_dir, f"seedv_{split}.npz") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            d = np.load(cache_path, allow_pickle=True)
            data, labels = d['data'], d['labels']                     # read each array once (every d[...] access decompresses it)
            self.samples = [(torch.from_numpy(data[i]), int(labels[i])) for i in range(len(labels))]
            self.metadata = [tuple(m) for m in d['metadata']]
            print(f"  [SEED-V {split}] {len(self.samples):,} windows from cache {cache_path}", flush=True)
            return
        for cnt_file in sorted(f for f in os.listdir(root) if f.endswith('.cnt')):
            parts = cnt_file.replace('.cnt', '').split('_')
            subject, session = int(parts[0]), int(parts[1])
            if subject not in SUBJECTS or cnt_file in EXCLUDED:
                continue
            try:
                raw = _load_cnt(os.path.join(root, cnt_file))
            except Exception as e:
                print(f"  skipping {cnt_file}: {e}", flush=True)
                continue
            upper = {ch.upper(): ch for ch in raw.ch_names}
            pick = [upper[ch.upper()] for ch in CHANNELS_62 if ch.upper() in upper]
            if len(pick) < 62:
                print(f"  skipping {cnt_file}: {len(pick)}/62 channels", flush=True)
                continue
            raw.pick_channels(pick, ordered=True)
            sfreq_orig = int(raw.info['sfreq'])
            nyq = sfreq_orig / 2.0
            sos = butter(5, [max(bandpass[0] / nyq, 0.001), min(bandpass[1] / nyq, 0.999)], btype='band', output='sos')
            b_n, a_n = iirnotch(notch_freq, Q=30, fs=sfreq_orig)
            g = gcd(sfreq_target, sfreq_orig)
            up, down = sfreq_target // g, sfreq_orig // g
            data = raw.get_data().astype(np.float64)
            data = filtfilt(b_n, a_n, data, axis=-1)
            data = sosfiltfilt(sos, data, axis=-1)
            if up != down:
                data = resample_poly(data, up, down, axis=-1)
            data = data.astype(np.float32)
            for t in trials:
                s0 = int(SESSION_START_TIMES[session][t] * sfreq_target)
                s1 = min(int(SESSION_END_TIMES[session][t] * sfreq_target), data.shape[1])
                trial = data[:, s0:s1]
                for w in range(trial.shape[1] // self.window_samples):
                    chunk = trial[:, w * self.window_samples:(w + 1) * self.window_samples]
                    if chunk.shape[1] != self.window_samples:
                        continue
                    self.samples.append((torch.from_numpy((chunk * scale).copy()), TRIAL_LABELS[session][t]))
                    self.metadata.append((subject, session, t, w))
        if cache_path and self.samples:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, data=np.stack([s[0].numpy() for s in self.samples]),
                                labels=np.array([s[1] for s in self.samples]), metadata=np.array(self.metadata))
        print(f"  [SEED-V {split}] {len(self.samples):,} windows, labels={sorted(Counter(s[1] for s in self.samples).items())}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

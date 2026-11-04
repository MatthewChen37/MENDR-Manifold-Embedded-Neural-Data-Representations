"""CHB-MIT seizure detection: 16 bipolar channels, 10 s windows at 200 Hz.

Input: the per-recording pickles written by the CHB-MIT `process1.py` of the BIOT repository (one dict per recording with the bipolar channels at
256 Hz in microvolts and metadata['times'] = seizure (start, end) samples). Each recording is resampled to 200 Hz and
scaled x1e-2 (microvolts -> volts x1e4, the unit of every other dataset). Non-seizure windows: non-overlapping 10 s
windows that do not touch a seizure. Seizure windows: 10 s windows at 5 s stride overlapping a seizure. Per recording
with seizures, non-seizure windows are thinned (evenly spaced) to at most 10 per seizure window.
Subjects chb01-chb19 train, chb20-chb21 validation, chb22-chb23 test (missing subjects are skipped).
"""
import os
import pickle
from collections import Counter
from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

BIPOLAR_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
]
SFREQ_ORIG, SFREQ = 256, 200
EPOCH_SAMPLES = SFREQ * 10
SPLITS = {"train": [f'chb{i:02d}' for i in range(1, 20)], "val": [f'chb{i:02d}' for i in range(20, 22)],
          "test": [f'chb{i:02d}' for i in range(22, 24)]}


class CHBMIT(Dataset):
    """`root` contains chbXX/ folders of .pkl recordings. metadata[i] = subject of sample i."""

    def __init__(self, root, split, cache_dir=None, scale=1e-2, max_nonseizure_ratio=10):
        self.samples, self.metadata = [], []
        cache_path = os.path.join(cache_dir, f"chbmit_{split}.npz") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            d = np.load(cache_path, allow_pickle=True)
            data, labels = d['data'], d['labels']                     # read each array once (every d[...] access decompresses it)
            self.samples = [(torch.from_numpy(data[i]), int(labels[i])) for i in range(len(labels))]
            self.metadata = [str(m) for m in d['metadata']]
            print(f"  [CHB-MIT {split}] {len(self.samples):,} windows from cache {cache_path}", flush=True)
            return
        g = gcd(SFREQ, SFREQ_ORIG)
        up, down = SFREQ // g, SFREQ_ORIG // g
        for subj in SPLITS[split]:
            subj_dir = os.path.join(root, subj)
            if not os.path.isdir(subj_dir):
                continue
            for pkl_file in sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl')):
                try:
                    with open(os.path.join(subj_dir, pkl_file), 'rb') as fh:
                        d = pickle.load(fh)
                except Exception as e:
                    print(f"  skipping {pkl_file}: {e}", flush=True)
                    continue
                if any(ch not in d for ch in BIPOLAR_CHANNELS):
                    continue
                x = resample_poly(np.stack([d[ch] for ch in BIPOLAR_CHANNELS], axis=0), up, down, axis=1).astype(np.float32)
                x = x * scale
                T = x.shape[1]
                seizures = [(int(s * SFREQ / SFREQ_ORIG), int(e * SFREQ / SFREQ_ORIG)) for s, e in d['metadata']['times']]
                seiz, non = [], []
                for w in range(T // EPOCH_SAMPLES):
                    a, b = w * EPOCH_SAMPLES, (w + 1) * EPOCH_SAMPLES
                    chunk = x[:, a:b]
                    if chunk.shape[1] == EPOCH_SAMPLES and not any(a < e and b > s for s, e in seizures):
                        non.append((torch.from_numpy(chunk.copy()), 0))
                for s, e in seizures:
                    a = max(0, s - EPOCH_SAMPLES + 1)
                    while a + EPOCH_SAMPLES <= T:
                        b = a + EPOCH_SAMPLES
                        if a < e and b > s:
                            chunk = x[:, a:b]
                            if chunk.shape[1] == EPOCH_SAMPLES:
                                seiz.append((torch.from_numpy(chunk.copy()), 1))
                        if a >= e:
                            break
                        a += EPOCH_SAMPLES // 2
                if max_nonseizure_ratio is not None and seiz:
                    max_ns = int(len(seiz) * max_nonseizure_ratio)
                    if len(non) > max_ns:
                        non = [non[i] for i in np.linspace(0, len(non) - 1, max_ns, dtype=int)]
                self.samples += seiz + non
                self.metadata += [subj] * (len(seiz) + len(non))
        if cache_path and self.samples:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, data=np.stack([s[0].numpy() for s in self.samples]),
                                labels=np.array([s[1] for s in self.samples]), metadata=np.array(self.metadata, dtype=str))
        print(f"  [CHB-MIT {split}] {len(self.samples):,} windows, labels={sorted(Counter(s[1] for s in self.samples).items())}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

"""PhysioNet EEG Motor Movement/Imagery (eegmmidb 1.0.0), 4-class motor imagery, 64 channels at 200 Hz.

Runs 4/8/12 (left vs right fist) and 6/10/14 (both fists vs both feet); one 4 s epoch from each T1/T2 cue onset
(labels: left 0, right 1, fists 2, feet 3). Per run: 60 Hz IIR notch (Q 30) at the file's own sampling rate, common
average reference, 5th-order Butterworth 0.1-75 Hz (zero phase), polyphase resampling to 200 Hz, scaling x1e4.
Evaluation: 5-fold cross-validation over subjects (mendr.tasks.physio_folds).
"""
import hashlib
import os
from collections import Counter
from math import gcd

import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from torch.utils.data import Dataset

mne.set_log_level('ERROR')

LR_RUNS = [4, 8, 12]
BF_RUNS = [6, 10, 14]
CHANNELS_64 = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
    'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
    'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
    'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.',
    'Tp7.', 'Tp8.',
    'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
    'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..', 'Iz..',
]
SUBJECTS = [f"S{i:03d}" for i in range(1, 110)]


class PhysioMI(Dataset):
    """All epochs of `subjects`; `subject_ids[i]` is the subject of sample i. `root` contains S001/ ... S109/."""

    def __init__(self, root, subjects=SUBJECTS, cache_dir=None, sfreq_target=200, duration=4.0, bandpass=(0.1, 75.0)):
        self.samples, self.subject_ids = [], []
        cache_path = None
        if cache_dir:
            key = hashlib.md5(f"{','.join(sorted(subjects))}|{sfreq_target}|{duration}|{bandpass}".encode()).hexdigest()[:16]
            cache_path = os.path.join(cache_dir, f"physio_{key}.npz")
            if os.path.exists(cache_path):
                d = np.load(cache_path, allow_pickle=True)
                data, labels = d['data'], d['labels']                 # read each array once (every d[...] access decompresses it)
                self.samples = [(torch.from_numpy(data[i]), int(labels[i])) for i in range(len(labels))]
                self.subject_ids = [str(s) for s in d['subjects']]
                print(f"  PhysioMI: {len(self.samples)} epochs from cache {cache_path}", flush=True)
                return
        T = int(sfreq_target * duration)
        for subj in subjects:
            subj_dir = os.path.join(root, subj)
            if not os.path.isdir(subj_dir):
                continue
            for run in LR_RUNS + BF_RUNS:
                fpath = os.path.join(subj_dir, f"{subj}R{run:02d}.edf")
                if not os.path.exists(fpath):
                    continue
                try:
                    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                    avail = [ch for ch in CHANNELS_64 if ch in raw.ch_names]
                    if len(avail) < 64:
                        continue
                    raw.pick_channels(avail, ordered=True)
                    events, _ = mne.events_from_annotations(raw, event_id={'T1': 2, 'T2': 3}, verbose=False)
                    sfreq_orig = int(round(raw.info['sfreq']))
                    g = gcd(sfreq_target, sfreq_orig)
                    up, down = sfreq_target // g, sfreq_orig // g
                    nyq = sfreq_orig / 2.0
                    sos = butter(5, [max(bandpass[0] / nyq, 0.001), min(bandpass[1] / nyq, 0.999)], btype='band', output='sos')
                    b_n, a_n = iirnotch(60.0, Q=30.0, fs=sfreq_orig)
                    raw_data = filtfilt(b_n, a_n, raw.get_data(), axis=-1).astype(np.float32)
                    raw_data = raw_data - raw_data.mean(axis=0, keepdims=True)
                    data = sosfiltfilt(sos, raw_data, axis=-1).astype(np.float32)
                    if up != down:
                        data = resample_poly(data, up, down, axis=1).astype(np.float32)
                    for onset, _, code in events:
                        if code not in [2, 3]:
                            continue
                        onset_rs = int(onset * sfreq_target / sfreq_orig)
                        label = (0 if code == 2 else 1) if run in LR_RUNS else (2 if code == 2 else 3)
                        epoch = data[:, onset_rs:onset_rs + T]
                        if epoch.shape[1] < T:
                            continue
                        self.samples.append((torch.from_numpy(epoch * 1e4), label))
                        self.subject_ids.append(subj)
                except Exception:
                    continue
        if cache_path and self.samples:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, data=np.stack([s[0].numpy() for s in self.samples]),
                                labels=np.array([s[1] for s in self.samples]), subjects=np.array(self.subject_ids))
        print(f"  PhysioMI: {len(self.samples)} epochs, labels={sorted(Counter(s[1] for s in self.samples).items())}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

"""Per-task settings of the paper's Table 1 and the data / model builders shared by finetune.py and evaluate.py."""
import random

import numpy as np
from torch.utils.data import DataLoader, Subset

from mendr.model import MENDRClassifier, load_pretrained_encoder, set_filter_bands
from mendr.train import collate_fn, weighted_sampler

COMMON = dict(lr=1e-4, wd=0.05, epochs=50, batch_size=64, dropout=0.1, label_smoothing=0.1, enc_lr_mult=0.5,
              cls_lr_mult=5.0, bn_perband_eval=50, d_model=128, n_transformer_layers=4, n_heads=4, d_temporal=64,
              n_temporal_windows=8)

# in_channels, classes, window/stride (samples at 200 Hz), temporal path version, augmentation, whether the pretrained
# channel projection is loaded (otherwise it starts from its fresh initialisation), filter-band plan applied after loading,
# no weight decay on the filter cutoffs, reported metric.
TASKS = {
    "tuab":   dict(in_channels=19, num_classes=2, window_samples=64, stride_samples=32, temporal_version=1, augment=True,
                   load_spatial_proj=False, filter_plan=None, nodecay_filters=False, metric="auroc"),
    "tuev":   dict(in_channels=19, num_classes=6, window_samples=128, stride_samples=64, temporal_version=1, augment=True,
                   load_spatial_proj=True, filter_plan=None, nodecay_filters=False, metric="f1"),
    "physio": dict(in_channels=64, num_classes=4, window_samples=128, stride_samples=64, temporal_version=1, augment=True,
                   load_spatial_proj=False, filter_plan=None, nodecay_filters=False, metric="f1"),
    "seedv":  dict(in_channels=62, num_classes=5, window_samples=200, stride_samples=200, temporal_version=2, augment=False,
                   load_spatial_proj=False, filter_plan="gamma4", nodecay_filters=True, metric="f1"),
    "chbmit": dict(in_channels=16, num_classes=2, window_samples=64, stride_samples=32, temporal_version=1, augment=False,
                   load_spatial_proj=False, filter_plan=None, nodecay_filters=False, metric="auroc"),
    "isruc":  dict(in_channels=6, num_classes=5, window_samples=64, stride_samples=32, temporal_version=1, augment=False,
                   load_spatial_proj=False, filter_plan=None, nodecay_filters=False, metric="f1"),
}


def config(task):
    return {**COMMON, **TASKS[task]}


def build_model(task):
    c = config(task)
    return MENDRClassifier(
        in_channels=c["in_channels"], num_classes=c["num_classes"], d_spatial=32, n_filters=4, sfreq=200,
        window_samples=c["window_samples"], stride_samples=c["stride_samples"], d_model=c["d_model"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_transformer_layers"], d_temporal=c["d_temporal"], dropout=c["dropout"],
        temporal_version=c["temporal_version"], n_temporal_windows=c["n_temporal_windows"])


def load_encoder(model, task, encoder_path, device):
    """Load the pretrained encoder into a fresh classifier as in Table 1 (and the task's filter plan, if any)."""
    c = config(task)
    loaded, _, modules = load_pretrained_encoder(model, encoder_path, device, skip_spatial_proj=not c["load_spatial_proj"])
    print(f"    Loaded {len(loaded)} pretrained tensors from {modules} (channel projection "
          f"{'loaded' if c['load_spatial_proj'] else 'freshly initialised'})", flush=True)
    if c["filter_plan"]:
        print(f"    filter bands set to '{c['filter_plan']}': {set_filter_bands(model, c['filter_plan'])}", flush=True)
    return loaded


def build_splits(task, data_root, cache_dir=None):
    """Datasets of a task. Returns (train, val, test) for every task except PhysioMI, which returns (dataset, folds)
    with folds = [(train_idx, val_idx, test_idx)] * 5."""
    if task in ("tuab", "tuev"):
        from mendr.data.tuh import build_tuh_task
        tr, va, te, _, _ = build_tuh_task(task, data_root, cache_dir=cache_dir)
        return tr, va, te
    if task == "physio":
        from mendr.data.physio import PhysioMI
        data = PhysioMI(data_root, cache_dir=cache_dir)
        return data, physio_folds(data.subject_ids)
    if task == "isruc":
        from mendr.data.isruc import ISRUC as D
    elif task == "seedv":
        from mendr.data.seedv import SEEDV as D
    elif task == "chbmit":
        from mendr.data.chbmit import CHBMIT as D
    else:
        raise ValueError(task)
    return tuple(D(data_root, split, cache_dir=cache_dir) for split in ("train", "val", "test"))


def physio_folds(subject_ids):
    """5-fold cross-validation over subjects (sorted, contiguous folds); inside each fold 15 % of the remaining subjects
    (a fixed shuffle per fold) form the validation set."""
    subj_list = sorted(set(subject_ids))
    folds = []
    for fi, test_subs in enumerate(np.array_split(subj_list, 5)):
        test_subs = set(test_subs)
        shuf = sorted(s for s in subj_list if s not in test_subs)
        random.Random(20260723 + fi).shuffle(shuf)
        n_val = max(1, int(round(len(shuf) * 0.15)))
        val_subs, tr_subs = set(shuf[:n_val]), set(shuf[n_val:])
        folds.append(tuple([i for i, s in enumerate(subject_ids) if s in part] for part in (tr_subs, val_subs, test_subs)))
    return folds


def loaders(task, train, val, test):
    """Training loader: class-balanced sampling (PhysioMI: plain shuffling); evaluation loaders in order."""
    bs = COMMON["batch_size"]
    if task == "physio":
        tr = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    else:
        tr = DataLoader(train, batch_size=bs, sampler=weighted_sampler(train), collate_fn=collate_fn, num_workers=0)
    return (tr, DataLoader(val, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0),
            DataLoader(test, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0))


def physio_fold_sets(data, fold):
    tr_idx, va_idx, te_idx = fold
    return Subset(data, tr_idx), Subset(data, va_idx), Subset(data, te_idx)

"""Fine-tuning and evaluation: linear probing, then full fine-tuning; model selection on the validation split; one test
evaluation of the selected weights."""
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler

from mendr.perband_bn import PerBandBN

PROBE_EPOCHS = 5          # head-only epochs when starting from a pretrained encoder
PATIENCE = 15             # early stopping on the validation metric


def configure_precision():
    """On GPU, use TF32 for float32 matrix products and convolutions: the setting of the paper's runs (the NVIDIA PyTorch
    container's default). Without it, GPU results differ slightly from the reported ones."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def weighted_sampler(dataset):
    """Class-balanced sampling with replacement (one epoch = len(dataset) draws)."""
    labels = [s[1] for s in dataset]
    counts = Counter(labels)
    return WeightedRandomSampler([1.0 / counts[lab] for lab in labels], num_samples=len(labels), replacement=True)


def class_weights(labels, n_classes, device):
    """sqrt(inverse class frequency), normalised to mean 1."""
    counts, total = Counter(labels), len(labels)
    cw = torch.ones(n_classes, device=device)
    for cls, cnt in counts.items():
        cw[cls] = (total / (n_classes * cnt)) ** 0.5
    return cw / cw.mean()


class BatchAugment:
    """Train-time augmentation of a batch [B, C, T]: random time shift of up to 5 % of the window (edges replicated, no
    wrap-around) and independent channel dropout with p = 0.1 (a sample never loses every channel)."""

    def __init__(self, max_shift_frac=0.05, p_channel_drop=0.1):
        self.max_shift_frac, self.p_channel_drop = max_shift_frac, p_channel_drop

    def __call__(self, x):
        B, C, T = x.shape
        S = int(round(self.max_shift_frac * T))
        if S > 0:
            s = torch.randint(-S, S + 1, (B, 1), device=x.device)
            idx = (torch.arange(T, device=x.device)[None, :] + s).clamp_(0, T - 1)
            x = x.gather(2, idx[:, None, :].expand(B, C, T))
        if self.p_channel_drop > 0:
            keep = torch.rand(B, C, device=x.device) >= self.p_channel_drop
            keep[~keep.any(dim=1)] = True
            x = x * keep[:, :, None].to(x.dtype)
        return x


def freeze_encoder(model, freeze=True):
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = not freeze


def make_optimizer(model, lr_encoder, lr_classifier, weight_decay, nodecay_filters=False):
    """AdamW with separate learning rates for the encoder and the head; with `nodecay_filters` the filter-bank cutoffs
    (in Hz) get no weight decay."""
    encoder_params, classifier_params, freq_params = [], [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        elif nodecay_filters and ('filter_bank.low_hz' in name or 'filter_bank.band_hz' in name):
            freq_params.append(param)
        else:
            encoder_params.append(param)
    groups = [{'params': encoder_params, 'lr': lr_encoder}, {'params': classifier_params, 'lr': lr_classifier}]
    if freq_params:
        groups.append({'params': freq_params, 'lr': lr_encoder, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, criterion, device, clip=1.0, augment=None):
    model.train()
    total_loss, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if augment is not None:
            x = augment(x)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        if torch.isnan(loss):
            continue
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(total_norm):
            optimizer.zero_grad()
            continue
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate_binary(model, loader, device):
    """AUROC / AUC-PR of softmax[:, 1]; balanced accuracy and weighted F1 at threshold 0.5."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
            labels.extend(y.numpy())
    labels = [int(l) for l in labels]
    preds = [1 if p > 0.5 else 0 for p in probs]
    from sklearn.metrics import average_precision_score
    m = {'bal_acc': balanced_accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}
    try:
        m['auroc'] = roc_auc_score(labels, probs)
    except ValueError:
        m['auroc'] = 0.0
    try:
        m['auc_pr'] = average_precision_score(labels, probs)
    except ValueError:
        m['auc_pr'] = 0.0
    return m


def evaluate_multi(model, loader, device):
    """'f1' is the weighted F1 (the reported metric); macro F1, balanced accuracy and Cohen's kappa are also returned."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return {'f1': f1_score(labels, preds, average='weighted'), 'weighted_f1': f1_score(labels, preds, average='weighted'),
            'macro_f1': f1_score(labels, preds, average='macro'), 'bal_acc': balanced_accuracy_score(labels, preds),
            'kappa': cohen_kappa_score(labels, preds)}


def per_class_eval(model, loader, device):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.extend(model(x.to(device)).argmax(1).cpu().numpy().tolist()); Y.extend(np.asarray(y).tolist())
    labels = sorted(set(Y) | set(P))
    return {"labels": [int(l) for l in labels],
            "f1_per_class": [float(v) for v in f1_score(Y, P, average=None, labels=labels)],
            "f1_weighted": float(f1_score(Y, P, average="weighted")), "f1_macro": float(f1_score(Y, P, average="macro")),
            "confusion": confusion_matrix(Y, P, labels=labels).tolist(),
            "support": [int(sum(1 for t in Y if t == l)) for l in labels]}


def finetune(model, train_loader, val_loader, test_loader, train_labels, cfg, device, num_classes, is_binary):
    """Train with selection on the validation split, then evaluate the selected weights on the test split once.

    cfg: dict with lr, wd, epochs, label_smoothing, enc_lr_mult, cls_lr_mult, pretrained, augment, bn_perband_eval,
    nodecay_filters, save_weights_to. Returns (best validation metrics, test metrics, history).

    Pretrained encoder: PROBE_EPOCHS epochs train the head only (lr x10), then everything is unfrozen with encoder lr
    lr*enc_lr_mult and head lr lr*cls_lr_mult under cosine decay. From scratch: lr / lr*10 from epoch 0, 3 warm-up epochs.
    """
    lr, wd, epochs, nd = cfg["lr"], cfg["wd"], cfg["epochs"], cfg.get("nodecay_filters", False)
    if cfg["pretrained"]:
        probe_epochs = PROBE_EPOCHS
        freeze_encoder(model, freeze=True)
        optimizer = make_optimizer(model, 0, lr * 10, wd, nd)
        print(f"    Linear probing for {probe_epochs} epochs (encoder frozen)", flush=True)
    else:
        probe_epochs = 0
        optimizer = make_optimizer(model, lr, lr * 10, wd, nd)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 3))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[3])
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, 2 if is_binary else num_classes, device),
                                    label_smoothing=cfg["label_smoothing"])

    augment = BatchAugment() if cfg.get("augment") else None
    eval_fn = evaluate_binary if is_binary else evaluate_multi
    metric_key = "auroc" if is_binary else "f1"
    _eval = lambda loader: eval_fn(model, loader, device)
    pbn = None
    if int(cfg.get("bn_perband_eval", 0) or 0) > 0:
        pbn = PerBandBN(model, int(cfg["bn_perband_eval"]), seed=0)
        print(f"    [per-band BN at evaluation] {pbn.describe()}", flush=True)
        _eval_plain = _eval

        def _eval(loader):
            pbn.calibrate(train_loader, device)
            with pbn.applied():
                return _eval_plain(loader)

    best_val_metric, best_state, best_val_metrics, patience_ctr, history = -1.0, None, {}, 0, []
    for epoch in range(epochs):
        if epoch == probe_epochs and probe_epochs > 0:
            freeze_encoder(model, freeze=False)
            optimizer = make_optimizer(model, lr * cfg["enc_lr_mult"], lr * cfg["cls_lr_mult"], wd, nd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - probe_epochs))
            print(f"    Unfreezing encoder at epoch {epoch} (lr={lr * cfg['enc_lr_mult']:.1e}/{lr * cfg['cls_lr_mult']:.1e})", flush=True)
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device, augment=augment)
        scheduler.step()
        val_metrics = _eval(val_loader)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        val_score = val_metrics.get(metric_key, val_metrics.get('bal_acc', 0.0))
        if val_score > best_val_metric:
            best_val_metric = val_score
            best_val_metrics = {**val_metrics, 'epoch': epoch, 'train_loss': loss}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if epoch % 5 == 0 or patience_ctr == 0:
            v_disp = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if isinstance(v, (int, float)))
            print(f"      ep {epoch:3d} | loss={loss:.4f} | val: {v_disp} ({time.time() - t0:.0f}s) "
                  f"{'** best-val' if patience_ctr == 0 else ''}", flush=True)
        if patience_ctr >= PATIENCE:
            print(f"      Early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if cfg.get("save_weights_to"):
            path = cfg["save_weights_to"]
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            assert not os.path.exists(path), f"refusing to overwrite {path}"
            torch.save(best_state, path)
            print(f"    saved best-val weights -> {path}", flush=True)
    test_metrics = evaluate_test(model, train_loader, test_loader, device, is_binary, pbn)
    print(f"    best_val_{metric_key}={best_val_metric:.4f} | test_{metric_key}={test_metrics.get(metric_key, 0):.4f}", flush=True)
    return best_val_metrics, test_metrics, history


def evaluate_test(model, train_loader, test_loader, device, is_binary, pbn=None):
    """Test metrics of the current weights. With per-band BN (`pbn`), the statistics are first recalibrated on the
    training split; this is also how released weights are evaluated (see evaluate.py)."""
    if pbn is not None:
        pbn.calibrate(train_loader, device); pbn.mode = "apply"
    try:
        metrics = (evaluate_binary if is_binary else evaluate_multi)(model, test_loader, device)
        if not is_binary:
            metrics["per_class"] = per_class_eval(model, test_loader, device)
    finally:
        if pbn is not None:
            pbn.mode = None
    return metrics

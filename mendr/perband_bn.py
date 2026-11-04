"""Per-band BatchNorm statistics for evaluation.

The temporal path is applied once per frequency band, so during training every BatchNorm layer in `model.temporal_path`
normalises each band with that band's own batch statistics, whereas a plain eval-mode BatchNorm would use a single running
estimate blended over the bands. `PerBandBN` keeps evaluation consistent with training: `calibrate` recomputes, for every
temporal-path BatchNorm layer, one mean and variance per band (cumulative average over `n_batches` training batches drawn
with a fixed seed), and inside `applied()` band k is normalised with band k's statistics.

Nothing in the model changes (no parameter or buffer is added; the statistics live on this helper). Training is
unaffected: the patched BatchNorm forward is the original one whenever the helper is idle, and the global random states
are restored after calibration. Because the statistics are a function of the current weights and a fixed batch set,
calibrating released weights on the same training split reproduces the statistics their test score was computed with.
"""
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class PerBandBN:
    def __init__(self, model, n_batches=50, seed=0):
        self.model, self.n_batches, self.seed = model, int(n_batches), int(seed)
        self.K = int(getattr(model, "n_filters", 4))
        self.tp = model.temporal_path
        self.bns = [m for m in self.tp.modules() if isinstance(m, nn.BatchNorm1d)]
        assert self.bns, "no BatchNorm1d in model.temporal_path"
        self.mode, self.band, self._calls = None, 0, 0
        self.stats = None                                    # per layer: (mean [K, C], var [K, C])
        for i, bn in enumerate(self.bns):
            bn._pb_orig_forward = bn.forward
            bn.forward = self._make_forward(i, bn)
        self._h1 = model.register_forward_pre_hook(self._reset_counter)
        self._h2 = self.tp.register_forward_pre_hook(self._next_band)

    def describe(self):
        return f"{len(self.bns)} temporal BN layers x {self.K} bands, {self.n_batches} training batches, seed {self.seed}"

    def _reset_counter(self, module, inp):
        self._calls = 0

    def _next_band(self, module, inp):
        self.band = self._calls % self.K
        self._calls += 1

    def _make_forward(self, i, bn):
        def fwd(x):
            if self.mode is None:
                return bn._pb_orig_forward(x)
            if self.mode == "collect":
                dims = [0] + list(range(2, x.dim()))
                m = x.mean(dim=dims); v = x.var(dim=dims, unbiased=True)
                acc = self._acc[i]
                acc[0][self.band] += m.detach().double(); acc[1][self.band] += v.detach().double(); acc[2][self.band] += 1
                return F.batch_norm(x, None, None, bn.weight, bn.bias, True, 0.0, bn.eps)
            mean, var = self.stats[i]
            return F.batch_norm(x, mean[self.band].to(x.dtype), var[self.band].to(x.dtype), bn.weight, bn.bias, False, 0.0, bn.eps)
        return fwd

    def _plain_loader(self, train_loader):
        g = torch.Generator(); g.manual_seed(self.seed)
        return DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True, generator=g,
                          collate_fn=train_loader.collate_fn, num_workers=0, drop_last=True)

    @torch.no_grad()
    def calibrate(self, train_loader, device):
        """Recompute the per-band statistics on `n_batches` batches of `train_loader.dataset` (shuffled with `seed`)."""
        st = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        was_training = self.model.training
        try:
            self.model.eval()
            dev = next(self.model.parameters()).device
            self._acc = [[torch.zeros(self.K, bn.num_features, dtype=torch.float64, device=dev),
                          torch.zeros(self.K, bn.num_features, dtype=torch.float64, device=dev),
                          torch.zeros(self.K, 1, dtype=torch.float64, device=dev)] for bn in self.bns]
            self.mode = "collect"
            for b, batch in enumerate(self._plain_loader(train_loader)):
                if b >= self.n_batches:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                self.model(x.to(device))
            self.stats = [((a[0] / a[2]).float(), (a[1] / a[2]).float()) for a in self._acc]
            assert all(float(a[2].min()) > 0 for a in self._acc), "a band received no calibration batch"
        finally:
            self.mode = None
            self.model.train(was_training)
            random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2])
            if st[3] is not None:
                torch.cuda.set_rng_state_all(st[3])

    @contextmanager
    def applied(self):
        assert self.stats is not None, "calibrate() first"
        self.mode = "apply"
        try:
            yield
        finally:
            self.mode = None

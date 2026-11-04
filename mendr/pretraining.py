"""MENDR self-supervised pretraining model.

Objectives:
  * masked SPD-token prediction under the Log-Euclidean distance (spatial path),
  * prediction of the Log-Euclidean Frechet mean of the masked tokens from the visible ones,
  * a filter-diversity penalty that keeps the learnable bands apart,
  * masked span reconstruction (MSR) for the temporal path: random spans of every band signal are zeroed, the temporal
    path's convolutions encode the masked signal, and a small convolutional decoder reconstructs the standardised signal
    on the masked samples. The MSR input is detached, so this loss trains only the temporal convolutions and the decoder.

`get_encoder_state_dict()` returns the weights that `mendr.model.load_pretrained_encoder` copies into a classifier.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from mendr.layers import SincFilterBank, WindowedCovarianceSPD, TemporalPath
from mendr.model import ChannelAgnosticSpatialProj, LogEuclideanEmbedding, SpatialTransformer


class MaskedSPDPretrainer(nn.Module):
    """Spatial-path pretraining (masked SPD prediction + Frechet mean + filter diversity).

    `forward` also computes a pooled-window reconstruction term for the temporal path (`temporal_decoder`), returned as
    the fourth value; `MENDRPretrainer` returns masked span reconstruction in that position instead.
    """

    def __init__(self, d_spatial=32, n_filters=4, sfreq=200, window_samples=64, stride_samples=32,
                 d_model=128, n_heads=4, n_transformer_layers=4, d_temporal=64, temporal_kernel=15, n_temporal_layers=2,
                 mask_ratio=0.5, dropout=0.1, ca_d_key=64, ca_n_heads=4):
        super().__init__()
        self.d_spatial = d_spatial
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.spatial_proj = ChannelAgnosticSpatialProj(d_spatial=d_spatial, d_key=ca_d_key, n_heads=ca_n_heads)
        self.filter_bank = SincFilterBank(n_filters=n_filters, kernel_size=65, sfreq=sfreq)
        self.windowed_cov = WindowedCovarianceSPD(window_samples, stride_samples)
        self.log_embed = LogEuclideanEmbedding(d_spatial)
        vec_dim = d_spatial * (d_spatial + 1) // 2
        self.spatial_transformer = SpatialTransformer(vec_dim=vec_dim, d_model=d_model, n_heads=n_heads,
                                                      n_layers=n_transformer_layers, dropout=dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, vec_dim) * 0.02)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vec_dim))
        self.frechet_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, vec_dim))
        self.temporal_path = TemporalPath(n_channels=d_spatial, d_temporal=d_temporal, kernel_size=temporal_kernel,
                                          n_layers=n_temporal_layers)
        self.temporal_decoder = nn.Sequential(nn.Linear(d_temporal + 1, d_model), nn.GELU(),
                                              nn.Linear(d_model, d_spatial * window_samples))
        self.window_samples = window_samples

    def encode_spatial(self, x):
        """x [B, C, T] -> (Log-Euclidean tokens [B, K*W, vec_dim], band signals [B, K, d_spatial, T])."""
        x_bands = self.filter_bank(self.spatial_proj(x))
        tokens = torch.cat([self.log_embed(self.windowed_cov(x_bands[:, k])) for k in range(x_bands.shape[1])], dim=1)
        return tokens, x_bands

    def forward(self, x):
        """Returns (spatial_loss, frechet_loss, temporal_loss, diversity_loss)."""
        B = x.shape[0]
        tokens, x_bands = self.encode_spatial(x)
        P, vec_dim = tokens.shape[1], tokens.shape[2]

        # masked SPD-token prediction
        n_mask = max(1, int(P * self.mask_ratio))
        n_visible = P - n_mask
        ids_shuffle = torch.rand(B, P, device=x.device).argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_visible, ids_masked = ids_shuffle[:, :n_visible], ids_shuffle[:, n_visible:]
        visible_tokens = tokens.gather(1, ids_visible.unsqueeze(-1).expand(-1, -1, vec_dim))
        masked_targets = tokens.detach().gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, vec_dim))
        full_tokens = torch.cat([visible_tokens, self.mask_token.expand(B, n_mask, -1)], dim=1)
        full_tokens = full_tokens.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, vec_dim))
        encoded = self.spatial_transformer.encode_tokens(full_tokens)
        pred_masked = self.decoder(encoded).gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, vec_dim))
        spatial_loss = ((pred_masked - masked_targets) ** 2).mean()                 # squared Log-Euclidean distance

        # Frechet mean of the masked tokens (arithmetic mean in the log domain) from the pooled visible tokens
        visible_encoded = encoded.gather(1, ids_visible.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        pred_frechet = self.frechet_head(visible_encoded.mean(dim=1))
        frechet_loss = ((pred_frechet - masked_targets.mean(dim=1)) ** 2).mean()

        # pooled-window temporal reconstruction (one random band)
        k = torch.randint(0, x_bands.shape[1], (1,)).item()
        band_signal = x_bands[:, k]
        T_sig = band_signal.shape[-1]
        temp_feat = self.temporal_path(band_signal)
        max_start = max(1, T_sig - self.window_samples)
        t_start = torch.randint(0, max_start, (B,), device=x.device)
        targets = []
        for i in range(B):
            window = band_signal[i, :, t_start[i]:t_start[i] + self.window_samples]
            if window.shape[-1] < self.window_samples:
                window = F.pad(window, (0, self.window_samples - window.shape[-1]))
            targets.append(window.reshape(-1))
        target_flat = torch.stack(targets)
        pos = (t_start.float() / max(max_start, 1)).unsqueeze(1)
        recon = self.temporal_decoder(torch.cat([temp_feat, pos], dim=1))
        temporal_loss = F.mse_loss(recon, target_flat)

        # filter diversity: centre frequencies at least 5 Hz apart
        fb = self.filter_bank
        low = fb.min_freq + torch.abs(fb.low_hz)
        centers = (low + (fb.min_band + torch.abs(fb.band_hz)) / 2).squeeze()
        n_f = centers.shape[0]
        dists = (centers.unsqueeze(0) - centers.unsqueeze(1)).abs()
        mask = 1.0 - torch.eye(n_f, device=x.device)
        diversity_loss = (torch.relu(5.0 - dists) * mask).sum() / max(n_f * (n_f - 1), 1)
        return spatial_loss, frechet_loss, temporal_loss, diversity_loss

    def get_encoder_state_dict(self):
        keep = ('spatial_proj', 'filter_bank', 'windowed_cov', 'log_embed', 'spatial_transformer', 'temporal_path')
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}

    def get_filter_frequencies(self):
        with torch.no_grad():
            fb = self.filter_bank
            low = fb.min_freq + torch.abs(fb.low_hz)
            high = torch.clamp(low + fb.min_band + torch.abs(fb.band_hz), max=fb.sfreq / 2.0)
        return list(zip(low.cpu().numpy(), high.cpu().numpy()))


class MENDRPretrainer(MaskedSPDPretrainer):
    """MENDR pretraining: the spatial objectives of the parent plus masked span reconstruction (MSR) for the temporal path.

    MSR zeroes `tpx_spans` spans of `tpx_span_len` samples (pairwise non-adjacent slots) in every band signal, encodes each
    band separately with `temporal_path.conv`, decodes every sample with `tpx_decoder` and scores masked samples against the
    signal standardised per (sample, band, channel) over time. The input is detached. Its BatchNorm calls use batch
    statistics without updating the running statistics. Span masks draw from their own generator, reseeded each epoch by
    `tpx_reseed`. The parent's forward runs unchanged; the MSR loss is returned as the temporal term.
    """

    def __init__(self, *args, tpx_spans=4, tpx_span_len=32, tpx_kernel=15, **kw):
        super().__init__(*args, **kw)
        d_t, d_s = self.temporal_path.d_temporal, self.d_spatial
        self.tpx_spans, self.tpx_span_len, self.tpx_kernel = int(tpx_spans), int(tpx_span_len), int(tpx_kernel)
        self.tpx_decoder = nn.Sequential(
            nn.Conv1d(d_t, d_t, self.tpx_kernel, padding=self.tpx_kernel // 2),
            nn.GELU(),
            nn.Conv1d(d_t, d_s, self.tpx_kernel, padding=self.tpx_kernel // 2),
        )
        self._bands = None
        self._tpx_seed, self._tpx_gen = None, None

    def tpx_reseed(self, seed):
        self._tpx_seed, self._tpx_gen = int(seed), None

    def _mask_generator(self, device):
        if self._tpx_seed is None:
            return None
        if self._tpx_gen is None or self._tpx_gen.device != torch.device(device):
            self._tpx_gen = torch.Generator(device=device)
            self._tpx_gen.manual_seed(self._tpx_seed)
        return self._tpx_gen

    def _capture(self, module, inputs, output):
        self._bands = output

    @contextlib.contextmanager
    def _bn_batch_stats_only(self):
        bns = [m for m in self.temporal_path.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        saved = [bn.track_running_stats for bn in bns]
        try:
            for bn in bns:
                bn.track_running_stats = False
            yield
        finally:
            for bn, s in zip(bns, saved):
                bn.track_running_stats = s

    def tpx_features(self, inp):
        """inp [B, K, C, T] -> [B*K, d_t, T], one temporal_path.conv call per band."""
        return torch.stack([self.temporal_path.conv(inp[:, k]) for k in range(inp.shape[1])], dim=1).flatten(0, 1)

    def span_mask(self, n, T, device):
        L, S = self.tpx_span_len, self.tpx_spans
        n_slots = (T - L + 1) // L
        assert n_slots >= 2 * S - 1, (T, L, S)
        g = self._mask_generator(device)
        off = torch.randint(0, T - n_slots * L + 1, (n, 1), device=device, generator=g)
        a = torch.rand(n, n_slots - S + 1, device=device, generator=g).argsort(dim=1)[:, :S].sort(dim=1).values
        slots = a + torch.arange(S, device=device)
        idx = (slots * L + off).unsqueeze(-1) + torch.arange(L, device=device)
        mask = torch.zeros(n, T, dtype=torch.bool, device=device)
        mask.scatter_(1, idx.reshape(n, -1), True)
        return mask

    def tpx_loss(self, x_bands, mask=None):
        xb = x_bands.detach()
        B, K, C, T = xb.shape
        flat = xb.reshape(B * K, C, T)
        mu = flat.mean(dim=-1, keepdim=True)
        sd = flat.std(dim=-1, keepdim=True).clamp_min(1e-12)
        target = (flat - mu) / sd
        if mask is None:
            mask = self.span_mask(B * K, T, xb.device)
        inp = (flat * (~mask).unsqueeze(1).to(flat.dtype)).reshape(B, K, C, T)
        with self._bn_batch_stats_only():
            pred = self.tpx_decoder(self.tpx_features(inp))
        sel = mask.unsqueeze(1).expand(-1, C, -1)
        return ((pred - target) ** 2)[sel].mean()

    def forward(self, x):
        handle = self.filter_bank.register_forward_hook(self._capture)
        try:
            spatial_loss, frechet_loss, _, diversity_loss = super().forward(x)
        finally:
            handle.remove()
        bands, self._bands = self._bands, None
        return spatial_loss, frechet_loss, self.tpx_loss(bands), diversity_loss

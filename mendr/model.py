"""MENDR classifier: channel-agnostic spatial projection, learnable filter bank, Log-Euclidean SPD tokens with a transformer
(spatial path) and a convolutional temporal path, fused by an MLP head."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mendr.layers import SincFilterBank, WindowedCovarianceSPD, TemporalPath, TemporalPathV2


class ChannelAgnosticSpatialProj(nn.Module):
    """Maps any number of input channels to `d_spatial` virtual electrodes by cross-attention.

    Each input channel is described by its mean amplitude and a learned embedding of its slot (channel order of the
    dataset); `d_spatial` learned queries attend to the channels, and the resulting softmax weights mix the raw signals:
    [B, C, T] -> [B, d_spatial, T]. `coord_encoder` and `_fourier_freqs` are an optional electrode-coordinate input that
    the released models do not use; they are kept so the checkpoints load with strict key matching.
    """

    def __init__(self, d_spatial=32, d_key=64, n_heads=4, max_channels=128):
        super().__init__()
        self.d_spatial = d_spatial
        self.d_key = d_key
        self.max_channels = max_channels
        self.virtual_electrodes = nn.Parameter(torch.randn(1, d_spatial, d_key) * 0.02)
        self.channel_pos_embed = nn.Parameter(torch.randn(max_channels, d_key) * 0.02)
        self.channel_encoder = nn.Sequential(nn.Linear(1 + d_key, d_key), nn.GELU(), nn.Linear(d_key, d_key))
        self.register_buffer("_fourier_freqs", 2.0 ** torch.arange(4).float() * math.pi)
        self.coord_encoder = nn.Sequential(nn.Linear(3 * 4 * 2, d_key), nn.GELU(), nn.Linear(d_key, d_key))
        self.cross_attn = nn.MultiheadAttention(d_key, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_key)
        self.out_norm = nn.LayerNorm(d_spatial)

    def _channel_positions(self, n_channels):
        if n_channels <= self.max_channels:
            return self.channel_pos_embed[:n_channels]
        pos = self.channel_pos_embed.unsqueeze(0).permute(0, 2, 1)
        return F.interpolate(pos, size=n_channels, mode='linear', align_corners=True).squeeze(0).permute(1, 0)

    def forward(self, x):
        B, C, T = x.shape
        ch_pos = self._channel_positions(C)
        x_with_pos = torch.cat([x.mean(dim=-1, keepdim=True), ch_pos.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        channel_feats = self.channel_encoder(x_with_pos)                           # [B, C, d_key]
        queries = self.virtual_electrodes.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, channel_feats, channel_feats)
        attn_out = self.norm(attn_out + queries)                                  # [B, d_spatial, d_key]
        mix = torch.bmm(attn_out, channel_feats.transpose(1, 2)) / (self.d_key ** 0.5)
        mix = F.softmax(mix, dim=-1)                                              # [B, d_spatial, C]
        out = torch.bmm(mix, x)
        return self.out_norm(out.permute(0, 2, 1)).permute(0, 2, 1)


def robust_eigh(A):
    """Symmetric eigendecomposition: symmetrise, break exact eigenvalue ties with a deterministic 1e-7-relative diagonal
    ramp, and fall back to CPU LAPACK if the batched GPU solver raises."""
    A = 0.5 * (A + A.transpose(-1, -2))
    n = A.shape[-1]
    scale = A.diagonal(dim1=-2, dim2=-1).abs().mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-30)
    A = A + 1e-7 * scale * torch.diag(torch.linspace(0.0, 1.0, n, device=A.device, dtype=A.dtype))
    try:
        return torch.linalg.eigh(A)
    except torch._C._LinAlgError:
        w, v = torch.linalg.eigh(A.detach().cpu())
        return w.to(A.device), v.to(A.device)


class _StableMatrixLog(torch.autograd.Function):
    """Matrix logarithm of SPD matrices (float64 eigendecomposition) with a Daleckii-Krein backward that is stable for
    near-degenerate eigenvalues. Output and gradient are returned in the caller's dtype."""

    @staticmethod
    def forward(ctx, spd):
        out_dtype = spd.dtype if spd.dtype in (torch.float32, torch.float64) else torch.float32
        eigenvalues, eigenvectors = robust_eigh(spd.to(torch.float64))
        eigenvalues = eigenvalues.clamp(min=1e-12)
        log_spd = eigenvectors @ torch.diag_embed(eigenvalues.log()) @ eigenvectors.transpose(-1, -2)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.out_dtype = out_dtype
        return log_spd.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        eigenvalues, eigenvectors = ctx.saved_tensors
        grad_output = grad_output.to(torch.float64)
        grad_sym = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        grad_eig = eigenvectors.transpose(-1, -2) @ grad_sym @ eigenvectors
        lam = eigenvalues
        lam_i, lam_j = lam.unsqueeze(-1), lam.unsqueeze(-2)
        diff_lam = lam_i - lam_j
        log_diff = lam_i.log() - lam_j.log()
        eps = 1e-6 * lam.abs().mean(dim=-1, keepdim=True).unsqueeze(-1)
        near_equal = diff_lam.abs() < eps
        safe_denom = torch.where(near_equal, torch.ones_like(diff_lam), diff_lam)
        kernel = torch.where(near_equal, 1.0 / lam_i.clamp(min=1e-12), log_diff / safe_denom)
        kernel = torch.where(torch.isfinite(kernel), kernel, 1.0 / lam_i.clamp(min=1e-12))
        grad_spd = eigenvectors @ (grad_eig * kernel) @ eigenvectors.transpose(-1, -2)
        grad_spd = 0.5 * (grad_spd + grad_spd.transpose(-1, -2))
        grad_spd = torch.where(torch.isfinite(grad_spd), grad_spd, torch.zeros_like(grad_spd))
        return grad_spd.to(ctx.out_dtype)


stable_matrix_log = _StableMatrixLog.apply


class LogEuclideanEmbedding(nn.Module):
    """SPD matrices [B, W, C, C] -> tangent vectors [B, W, C(C+1)/2]: matrix logarithm, upper triangle, off-diagonal
    entries scaled by sqrt(2) so the Euclidean distance of the vectors is the Log-Euclidean distance of the matrices."""

    def __init__(self, spd_dim):
        super().__init__()
        self.spd_dim = spd_dim
        self.vec_dim = spd_dim * (spd_dim + 1) // 2
        row_idx, col_idx = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer('row_idx', row_idx)
        self.register_buffer('col_idx', col_idx)

    def forward(self, spd):
        vec = stable_matrix_log(spd)[:, :, self.row_idx, self.col_idx]
        diag_mask = (self.row_idx == self.col_idx).float()
        scale = torch.where(diag_mask.bool(), torch.ones_like(diag_mask), math.sqrt(2) * torch.ones_like(diag_mask))
        return vec * scale.unsqueeze(0).unsqueeze(0)


class SpatialTransformer(nn.Module):
    """Transformer over the SPD token sequence with a learned positional embedding and attention pooling."""

    MAX_TOKENS = 2000

    def __init__(self, vec_dim, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(vec_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.MAX_TOKENS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout,
                                           activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def encode_tokens(self, tokens):
        """[B, W, vec_dim] -> [B, W, d_model]."""
        B, W, _ = tokens.shape
        assert W <= self.MAX_TOKENS, f"{W} tokens exceed MAX_TOKENS {self.MAX_TOKENS}"
        tokens = self.input_norm(self.input_proj(tokens)) + self.pos_embed[:, :W, :]
        return self.transformer(tokens)

    def forward(self, tokens):
        """[B, W, vec_dim] -> [B, d_model]."""
        encoded = self.encode_tokens(tokens)
        query = self.attn_pool_query.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.attn_pool(query, encoded, encoded)
        return pooled.squeeze(1)


class MENDRClassifier(nn.Module):
    """MENDR for classification. x [B, C, T] (any C) -> logits [B, num_classes].

    Spatial path: per band, windowed covariance -> Log-Euclidean tokens; the K bands' token sequences are concatenated
    and encoded by one transformer. Temporal path: applied to each band signal, features averaged over bands. The two
    are concatenated and classified by an MLP (`classifier`, the head trained during linear probing).
    """

    def __init__(self, in_channels, num_classes, d_spatial=32, n_filters=4, sfreq=200,
                 window_samples=64, stride_samples=32, d_model=128, n_heads=4, n_transformer_layers=4,
                 d_temporal=64, temporal_kernel=15, n_temporal_layers=2, dropout=0.1,
                 ca_d_key=64, ca_n_heads=4, temporal_version=1, n_temporal_windows=8):
        super().__init__()
        self.in_channels = in_channels
        self.d_spatial = d_spatial
        self.n_filters = n_filters
        self.d_model = d_model
        self.d_temporal = d_temporal
        self.spatial_proj = ChannelAgnosticSpatialProj(d_spatial=d_spatial, d_key=ca_d_key, n_heads=ca_n_heads)
        self.filter_bank = SincFilterBank(n_filters=n_filters, kernel_size=65, sfreq=sfreq)
        self.windowed_cov = WindowedCovarianceSPD(window_samples, stride_samples)
        self.log_embed = LogEuclideanEmbedding(d_spatial)
        self.spatial_transformer = SpatialTransformer(vec_dim=d_spatial * (d_spatial + 1) // 2, d_model=d_model,
                                                      n_heads=n_heads, n_layers=n_transformer_layers, dropout=dropout)
        if temporal_version == 2:
            self.temporal_path = TemporalPathV2(n_channels=d_spatial, d_temporal=d_temporal, n_temporal_windows=n_temporal_windows)
        else:
            self.temporal_path = TemporalPath(n_channels=d_spatial, d_temporal=d_temporal, kernel_size=temporal_kernel,
                                              n_layers=n_temporal_layers)
        fusion_dim = d_model + d_temporal
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x):
        x_bands = self.filter_bank(self.spatial_proj(x))                          # [B, K, d_spatial, T]
        K = x_bands.shape[1]
        tokens = torch.cat([self.log_embed(self.windowed_cov(x_bands[:, k])) for k in range(K)], dim=1)
        spatial = self.spatial_transformer(tokens)
        temporal = torch.stack([self.temporal_path(x_bands[:, k]) for k in range(K)], dim=1).mean(dim=1)
        return self.classifier(torch.cat([spatial, temporal], dim=1))

    def get_filter_frequencies(self):
        with torch.no_grad():
            fb = self.filter_bank
            low = fb.min_freq + torch.abs(fb.low_hz)
            high = torch.clamp(low + fb.min_band + torch.abs(fb.band_hz), max=fb.sfreq / 2.0)
        return list(zip(low.cpu().numpy(), high.cpu().numpy()))


def load_pretrained_encoder(model, encoder_path, device='cpu', skip_spatial_proj=False):
    """Copy a pretrained encoder (pretrain.py output) into a MENDRClassifier.

    `filter_bank.window` / `filter_bank.n_` are fixed buffers of the target model and are not copied. With
    `skip_spatial_proj` the channel projection keeps its fresh initialisation (the default for downstream tasks whose
    montage differs from pretraining). Returns (loaded keys, skipped keys, loaded top-level modules).
    """
    state = torch.load(encoder_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    skip_buffers = {'filter_bank.n_', 'filter_bank.window'}
    loaded, skipped = [], []
    for k, v in state.items():
        if k in skip_buffers or (skip_spatial_proj and k.startswith('spatial_proj.')):
            skipped.append(k)
        elif k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            loaded.append(k)
        elif k in model_state and 'pos_embed' in k and v.ndim == model_state[k].ndim:   # shorter positional table
            model_state[k][:, :v.shape[1], :] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped, sorted(set(k.split('.')[0] for k in loaded))


FILTER_PLANS = {"gamma4": [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]}


def set_filter_bands(model, plan):
    """Set the filter bank's cutoffs to a named plan (e.g. 'gamma4' = 4-8 / 8-13 / 13-30 / 30-45 Hz). Deterministic."""
    fb = model.filter_bank
    ref = SincFilterBank(n_filters=fb.n_filters, kernel_size=fb.kernel_size, sfreq=fb.sfreq, min_freq=fb.min_freq,
                         min_band=fb.min_band, bands=FILTER_PLANS[plan])
    with torch.no_grad():
        fb.low_hz.copy_(ref.low_hz.to(fb.low_hz.device))
        fb.band_hz.copy_(ref.band_hz.to(fb.band_hz.device))
    return [(round(float(lo), 2), round(float(hi), 2)) for lo, hi in model.get_filter_frequencies()]

"""Building blocks of MENDR: learnable sinc filter bank, windowed covariance (SPD) tokens and the temporal paths."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincFilterBank(nn.Module):
    """Learnable band-pass filter bank (SincNet-style).

    Filter i passes [low_i, high_i] with low_i = min_freq + |low_hz_i| and high_i = low_i + min_band + |band_hz_i|
    (capped at Nyquist). `bands` optionally gives the intended (low, high) edges in Hz; the raw parameters are then
    solved so the realised pass-bands equal them exactly.
    """

    def __init__(self, n_filters=4, kernel_size=65, sfreq=200, min_freq=1.0, min_band=2.0, bands=None):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sfreq = sfreq
        self.min_freq = min_freq
        self.min_band = min_band

        nyquist = sfreq / 2.0
        if bands is not None:
            if len(bands) != n_filters:
                raise ValueError(f"{len(bands)} bands given for n_filters={n_filters}")
            for lo, hi in bands:
                if lo < min_freq - 1e-6 or (hi - lo) < min_band - 1e-6 or hi > nyquist + 1e-6:
                    raise ValueError(f"band ({lo}, {hi}) is not representable at sfreq={sfreq}")
            init_low = [lo - min_freq for lo, _ in bands]
            init_band = [(hi - lo) - min_band for lo, hi in bands]
        elif n_filters == 4:
            init_low = [1.0, 4.0, 8.0, 13.0]
            init_band = [3.0, 4.0, 5.0, 17.0]
        else:
            raise ValueError("give `bands` for n_filters != 4")

        self.low_hz = nn.Parameter(torch.tensor(init_low, dtype=torch.float32))
        self.band_hz = nn.Parameter(torch.tensor(init_band, dtype=torch.float32))

        n = torch.arange(0, kernel_size).float()
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))     # Hamming
        self.register_buffer('window', window)
        n_ = (2 * math.pi * torch.arange(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0 + 1).float() / sfreq)
        self.register_buffer('n_', n_)

    def _sinc(self, x):
        mask = x == 0
        x = torch.where(mask, torch.ones_like(x), x)
        return torch.where(mask, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x):
        """x [B, C, T] -> [B, n_filters, C, T]."""
        low = self.min_freq + torch.abs(self.low_hz)
        high = torch.clamp(low + self.min_band + torch.abs(self.band_hz), max=self.sfreq / 2.0)
        filters = []
        for i in range(self.n_filters):
            low_pass1 = 2 * high[i] * self._sinc(self.n_ * high[i])
            low_pass2 = 2 * low[i] * self._sinc(self.n_ * low[i])
            bp = (low_pass1 - low_pass2) * self.window
            bp = bp / (bp.abs().sum() + 1e-8)
            filters.append(bp)
        filt = torch.stack(filters).unsqueeze(1)                                   # [K, 1, kernel]

        B, C, T = x.shape
        out = F.conv1d(x.reshape(B * C, 1, T), filt, padding=self.kernel_size // 2)   # [B*C, K, T]
        return out.reshape(B, C, self.n_filters, T).permute(0, 2, 1, 3)


class WindowedCovarianceSPD(nn.Module):
    """x [B, C, T] -> sequence of trace-normalised covariance matrices [B, W, C, C] (window, stride in samples).

    Each window's covariance is divided by its trace and regularised with 1e-3 * I, so the tokens are SPD and scale-free.
    """

    RIDGE = 1e-3

    def __init__(self, window_samples=64, stride_samples=32):
        super().__init__()
        self.window_samples = window_samples
        self.stride_samples = stride_samples

    def forward(self, x):
        B, C, T = x.shape
        win = min(self.window_samples, T)
        stride = min(self.stride_samples, win)
        windows = x.unfold(2, win, stride).permute(0, 2, 1, 3)                    # [B, W, C, win]
        windows = windows - windows.mean(dim=-1, keepdim=True)
        covs = torch.matmul(windows, windows.transpose(-1, -2)) / max(win - 1, 1)
        trace = covs.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        covs = covs / trace.clamp(min=1e-30)
        eye = torch.eye(C, device=x.device).reshape(1, 1, C, C)
        return covs + self.RIDGE * eye


class TemporalPath(nn.Module):
    """Temporal path (version 1): two convolution blocks over time, then attention pooling. [B, C, T] -> [B, d_temporal]."""

    def __init__(self, n_channels, d_temporal=64, kernel_size=15, n_layers=2):
        super().__init__()
        self.d_temporal = d_temporal
        layers = []
        for i in range(n_layers):
            in_ch = n_channels if i == 0 else d_temporal
            layers.extend([
                nn.Conv1d(in_ch, d_temporal, kernel_size, padding=kernel_size // 2, groups=1 if i == 0 else d_temporal),
                nn.BatchNorm1d(d_temporal),
                nn.GELU(),
            ])
        self.conv = nn.Sequential(*layers)
        self.attn_weights = nn.Sequential(
            nn.Linear(d_temporal, d_temporal // 2),
            nn.Tanh(),
            nn.Linear(d_temporal // 2, 1),
        )

    def forward(self, x):
        feat_t = self.conv(x).transpose(1, 2)                                      # [B, T, d_temporal]
        attn = F.softmax(self.attn_weights(feat_t), dim=1)
        return (feat_t * attn).sum(dim=1)


class TemporalPathV2(nn.Module):
    """Temporal path (version 2, used for SEED-V): multi-scale convolutions, mean-pooling into `n_temporal_windows`
    segments, a small transformer over the segment sequence and attention pooling. [B, C, T] -> [B, d_temporal]."""

    def __init__(self, n_channels, d_temporal=64, n_layers=2, n_temporal_windows=8, n_heads=4, n_transformer_layers=2):
        super().__init__()
        self.d_temporal = d_temporal
        self.n_temporal_windows = n_temporal_windows
        self.conv_small = nn.Sequential(
            nn.Conv1d(n_channels, d_temporal // 2, kernel_size=7, padding=3), nn.BatchNorm1d(d_temporal // 2), nn.GELU())
        self.conv_medium = nn.Sequential(
            nn.Conv1d(n_channels, d_temporal // 4, kernel_size=31, padding=15), nn.BatchNorm1d(d_temporal // 4), nn.GELU())
        self.conv_large = nn.Sequential(
            nn.Conv1d(n_channels, d_temporal // 4, kernel_size=65, padding=32), nn.BatchNorm1d(d_temporal // 4), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_temporal, d_temporal, kernel_size=15, padding=7, groups=d_temporal),
            nn.BatchNorm1d(d_temporal),
            nn.GELU(),
            nn.Conv1d(d_temporal, d_temporal, kernel_size=1),
            nn.BatchNorm1d(d_temporal),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_temporal_windows, d_temporal) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_temporal, nhead=n_heads, dim_feedforward=d_temporal * 2, dropout=0.1,
                                           activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_temporal) * 0.02)
        self.attn_pool = nn.MultiheadAttention(d_temporal, num_heads=1, batch_first=True)

    def forward(self, x):
        B, C, T = x.shape
        feat = torch.cat([self.conv_small(x), self.conv_medium(x), self.conv_large(x)], dim=1)
        feat = self.conv2(feat)                                                     # [B, d_temporal, T]
        W = self.n_temporal_windows
        seg_len = T // W
        if seg_len > 0 and T >= W:
            feat_windows = feat[:, :, :seg_len * W].reshape(B, self.d_temporal, W, seg_len).mean(dim=-1)
        else:
            feat_windows = F.adaptive_avg_pool1d(feat, W)
        feat_windows = feat_windows.transpose(1, 2) + self.pos_embed[:, :W, :]
        encoded = self.transformer(feat_windows)
        query = self.attn_pool_query.expand(B, -1, -1)
        pooled, _ = self.attn_pool(query, encoded, encoded)
        return pooled.squeeze(1)

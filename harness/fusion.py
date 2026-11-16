"""Input-image construction for NeuroSeg-III (Wu et al., Biomed Opt Express 15(5):2910, 2024).

Sec 2.2: the network input is the maximum projection linearly fused 1:1 with a
correlation map, then normalised. The correlation map (eq. 1) is the
Gaussian-weighted multi-dimensional correlation of each pixel with its neighbours:

    cw(f1..fn) = || sum_i g_i f_i ||^2 / ( sum_i g_i ||f_i||^2 )

With per-pixel z-scored traces (sum_t z^2 = T) and a normalised Gaussian kernel G,
this reduces to  corr(x) = sum_t (G * z_t)(x)^2 / T,  which is 1 for a pixel
perfectly correlated with its neighbourhood and ~sum(g^2) for independent noise.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def correlation_map(mov, sigma=1.0, chunk=250, eps=1e-6):
    """mov: (T,H,W). Returns float32 (H,W) local-correlation map."""
    T = mov.shape[0]
    mean = mov.mean(0, dtype=np.float64)
    var = np.zeros_like(mean)
    for s in range(0, T, chunk):
        var += ((mov[s:s + chunk].astype(np.float64) - mean) ** 2).sum(0)
    std = np.sqrt(var / T) + eps
    acc = np.zeros(mean.shape, np.float64)
    for s in range(0, T, chunk):
        z = (mov[s:s + chunk].astype(np.float32) - mean) / std
        g = gaussian_filter(z, sigma=(0, sigma, sigma), mode='nearest')
        acc += (g.astype(np.float64) ** 2).sum(0)
    return (acc / T).astype(np.float32)


def _norm01(x):
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.9)
    if hi <= lo: hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def fused_image(mov, sigma=1.0):
    """1:1 fusion of max projection and correlation map -> uint8 (H,W)."""
    mp = mov.max(0).astype(np.float32)
    cm = correlation_map(mov, sigma=sigma)
    return (( _norm01(mp) + _norm01(cm) ) / 2.0 * 255).astype(np.uint8)

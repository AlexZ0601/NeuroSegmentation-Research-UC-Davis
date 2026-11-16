"""Compat shims for running the vendored 2024-era ultralytics on a modern stack.

1. PyTorch >= 2.6 defaults torch.load(weights_only=True); the vendored loader
   calls torch.load() bare. Restore the old default (local, trusted checkpoints).
2. NumPy 2.0 removed np.trapz (renamed np.trapezoid); ultralytics/utils/metrics.py
   still calls it when computing AP, which crashes validation as soon as the model
   produces its first true positives.
"""
import numpy as np
import torch

if not getattr(torch.load, '_neuroseg_patched', False):
    _orig = torch.load

    def load(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)

    load._neuroseg_patched = True
    torch.load = load

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

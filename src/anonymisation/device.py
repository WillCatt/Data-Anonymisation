"""
Compute device detection.

Picks the best available PyTorch device at runtime: CUDA if a GPU is present,
Apple Silicon's MPS backend if on a Mac with M-series, otherwise CPU.

Importing torch is gated behind the function call so the rest of the
package still imports on a Phase-1-only install.
"""
from __future__ import annotations

from typing import Tuple


def best_device() -> Tuple[str, str]:
    """
    Return (device_name, friendly_label).

    Examples
    --------
    >>> best_device()
    ('cuda', 'NVIDIA GPU (CUDA)')        # on a Colab T4 or similar
    ('mps',  'Apple Silicon (MPS)')      # on M1/M2/M3 macOS
    ('cpu',  'CPU')                      # fallback
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This backend requires PyTorch. Install via `pip install -e \".[research]\"`."
        ) from exc

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", f"NVIDIA GPU (CUDA) — {gpu_name}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    return "cpu", "CPU (no GPU detected — this will be slow for fine-tuning)"


def report_device() -> str:
    """Pretty-print device info; useful as the first thing a notebook does."""
    device, label = best_device()
    return f"Device: {device}    ({label})"

"""Band normalization shared by inference and Grad-CAM.

The EfficientNet patches in ``data/datasets/efficientnet_dataset`` store
Sentinel-2 L2A reflectance on a 0-10,000 scale that was stretched to 0-1 with a
per-image, per-band 2/98 percentile stretch. Inference has to apply the same
transform to a whole scene: Sentinel-2 reflectance values are in the thousands,
so clipping them straight to ``0..1`` saturates every band and the classifier
only ever sees flat input.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

BAND_NAMES = ("B3", "B4", "B5", "B8", "B11")
LOW_PERCENTILE = 2.0
HIGH_PERCENTILE = 98.0


def valid_pixel_mask(array: np.ndarray) -> np.ndarray:
    """Pixels with finite values in at least one band.

    Outside-coverage pixels from a cloud-masked composite are exported as zeros,
    so a pixel where every band is zero is treated as missing rather than as
    extremely dark water.
    """
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    has_signal = np.any(finite & (array != 0.0), axis=-1)
    return finite.all(axis=-1) & has_signal


def stretch_band(band: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile-stretch one band to ``0..1``; invalid pixels become ``0``."""
    band = np.asarray(band, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(band)
    stretch = np.zeros_like(band, dtype=np.float32)
    if not np.any(valid):
        return stretch
    values = band[valid]
    low = float(np.percentile(values, LOW_PERCENTILE))
    high = float(np.percentile(values, HIGH_PERCENTILE))
    if high > low:
        scaled = (band - low) / (high - low)
    else:
        scaled = band - low
    stretch[valid] = np.clip(scaled[valid], 0.0, 1.0)
    return stretch


def normalize_bands(bands: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """Return a ``(height, width, 5)`` float32 array on the training scale.

    Accepts either a mapping of band name to 2-D array or a stacked
    ``(height, width, channels)`` array ordered as :data:`BAND_NAMES`.
    """
    if isinstance(bands, Mapping):
        stack = np.stack([np.asarray(bands[name], dtype=np.float32) for name in BAND_NAMES], axis=-1)
    else:
        stack = np.asarray(bands, dtype=np.float32)
        if stack.ndim != 3 or stack.shape[-1] != len(BAND_NAMES):
            raise ValueError(f"Expected a (height, width, {len(BAND_NAMES)}) stack, got {stack.shape}")

    valid = valid_pixel_mask(stack)
    normalized = np.zeros_like(stack, dtype=np.float32)
    for channel in range(stack.shape[-1]):
        normalized[..., channel] = stretch_band(stack[..., channel], valid)
    return normalized

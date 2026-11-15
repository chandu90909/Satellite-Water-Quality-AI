"""Patch helpers shared by classification and Grad-CAM."""
import numpy as np

PATCH_SIZE = 128


def pad_to_size(patch, size=PATCH_SIZE):
    """Zero-pad a ``(height, width, channels)`` patch up to ``size x size``.

    Images are not exact multiples of the 128 px tile grid, so the last row and
    column of tiles are smaller than the patch size the classifier was trained
    on. Padding with reflectance ``0`` keeps the model input shape fixed; the
    padded area is never written back into the output map or the explanation.
    """
    array = np.asarray(patch)
    if array.shape[0] == size and array.shape[1] == size:
        return array
    if array.shape[0] > size or array.shape[1] > size:
        raise ValueError(f"Patch of shape {array.shape} exceeds the {size}px tile size")
    padded = np.zeros((size, size, array.shape[2]), dtype=array.dtype)
    padded[: array.shape[0], : array.shape[1], :] = array
    return padded

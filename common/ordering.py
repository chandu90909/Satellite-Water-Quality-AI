"""Ordering helpers for acquisition names such as ``Hussain_Sagar_7.tif``.

Plain string sorting puts ``_10`` before ``_2``, which silently reorders every
temporal output. These helpers order by the trailing acquisition number instead.
"""
from pathlib import Path


def acquisition_number(name):
    """Return the trailing acquisition number, or -1 when the name has none."""
    stem = Path(str(name)).stem
    _, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return int(suffix)
    return -1


def acquisition_sort_key(name):
    """Sort key that orders ``Hussain_Sagar_2`` before ``Hussain_Sagar_10``."""
    stem = Path(str(name)).stem
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return (prefix, int(suffix), stem)
    return (stem, -1, stem)


def sort_acquisitions(items):
    """Return ``items`` (paths or names) in acquisition order."""
    return sorted(items, key=lambda item: acquisition_sort_key(getattr(item, "name", item)))

"""Internal HDF5 (en|de)coding helpers shared by save / load / validate.

These helpers convert between Python-side values (``str``, ``int``, ``float``,
``bool``, ``tuple``, ``list``, ``np.ndarray``) and the limited set of types
``h5py`` round-trips through file attributes. They live here so that
:meth:`TrajectoryDataset.load_hdf5` and :func:`validate` decode attributes
identically (no ``bytes``-vs-``str`` skew, no NumPy-scalar leakage).
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def encode_attr(value: Any) -> Any:
    """Convert ``value`` into an HDF5-attribute-friendly form.

    Tuples are coerced to lists so h5py stores them as arrays of homogeneous
    type; everything else is passed through and ``h5py`` decides. Callers that
    need stricter validation should do it before calling.
    """
    if isinstance(value, tuple):
        return list(value)
    return value


def decode_attr(value: Any) -> Any:
    """Inverse of :func:`encode_attr` for values read out of ``h5py``.

    - ``bytes`` → ``str`` (utf-8, ``errors="replace"``).
    - 0-d NumPy arrays / scalars → native Python scalars.
    - 1-D NumPy arrays → list (with element-wise byte→str decoding).
    - Everything else is returned unchanged.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_attr(value.item())
        return [decode_attr(v) for v in value.tolist()]
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    if isinstance(value, list):
        return [decode_attr(v) for v in value]
    return value


def read_group_attrs(group: h5py.Group | None) -> dict[str, Any]:
    """Read ``group.attrs`` into a plain ``dict`` with decoded values.

    Returns an empty dict when ``group`` is ``None`` so callers can do
    ``read_group_attrs(f.get("physical"))`` without branching on presence.
    """
    if group is None:
        return {}
    return {k: decode_attr(group.attrs[k]) for k in group.attrs}


def write_group_attrs(group: h5py.Group, values: dict[str, Any]) -> None:
    """Write ``values`` to ``group.attrs``, applying :func:`encode_attr` first."""
    for k, v in values.items():
        group.attrs[k] = encode_attr(v)


__all__ = ["decode_attr", "encode_attr", "read_group_attrs", "write_group_attrs"]

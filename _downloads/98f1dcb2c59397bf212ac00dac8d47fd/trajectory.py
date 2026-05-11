"""TrajectoryDataset — canonical container for trajectory data.

Holds a tensor of shape ``(n_traj, n_time, n_channel, *grid_shape)`` plus a
shared time grid, a compulsory dynamics identifier, and three open-content
metadata dicts grouped by meaning (``physical`` / ``numerical`` /
``provenance``). Every model and every interpretability extractor consumes
this object — see ``docs/ARCHITECTURE.md``.

Geometry (``Lx``, ``Nx``, ``dx``, …) is no longer a dedicated tensor; it
travels as scalar entries inside ``physical`` and ``numerical`` so the
container itself is geometry-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor

from ._hdf5 import decode_attr, encode_attr, read_group_attrs, write_group_attrs
from .format import CURRENT_TRAJECTORY_CACHE_FORMAT, FORMAT_NAME


@dataclass
class TrajectoryDataset:
    """A bundle of PDE trajectories with an explicit channel axis.

    Attributes:
        data: trajectories, shape ``(n_traj, n_time, n_channel, *grid_shape)``.
        times: shared time grid, shape ``(n_time,)``.
        dyn: compulsory dynamics identifier (registry key, e.g. ``"allen_cahn_1d"``).
        physical: equation/domain parameters (``Lx``, ``T``, ``eps``, …).
        numerical: discretization parameters (``Nx``, ``dx``, ``dt``, …).
        provenance: how the dataset was produced (``seed``, ``source``, …).
        channel_names: optional ordered channel labels; length must match
            ``data.shape[2]`` when provided. Single-field equations use ``("u",)``.
    """

    data: Tensor
    times: Tensor
    dyn: str
    physical: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dyn, str) or not self.dyn:
            raise ValueError(f"dyn must be a non-empty string; got {self.dyn!r}")
        if self.data.ndim < 4:
            raise ValueError(
                f"data must have shape (n_traj, n_time, n_channel, *grid_shape); got {tuple(self.data.shape)}"
            )
        n_time = self.data.shape[1]
        if tuple(self.times.shape) != (n_time,):
            raise ValueError(f"times shape {tuple(self.times.shape)} != ({n_time},)")
        if self.channel_names is not None:
            n_channel = self.data.shape[2]
            if len(self.channel_names) != n_channel:
                raise ValueError(f"channel_names length {len(self.channel_names)} != n_channel {n_channel}")
            self.channel_names = tuple(self.channel_names)

    @property
    def n_traj(self) -> int:
        return self.data.shape[0]

    @property
    def n_time(self) -> int:
        return self.data.shape[1]

    @property
    def n_channel(self) -> int:
        return self.data.shape[2]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[3:])

    @property
    def metadata(self) -> dict[str, Any]:
        """Flat read-only view bundling ``dyn`` and the three metadata groups.

        Convenience for one-shot inspection; downstream code should prefer the
        explicit ``physical`` / ``numerical`` / ``provenance`` dicts.
        """
        return {
            "dyn": self.dyn,
            "channel_names": self.channel_names,
            "physical": dict(self.physical),
            "numerical": dict(self.numerical),
            "provenance": dict(self.provenance),
        }

    def save_hdf5(self, path: str | Path) -> None:
        """Write the dataset to a v1 HDF5 cache file.

        Layout: ``data`` and ``times`` at the root; closed root attrs
        ``format`` / ``version`` / ``dyn`` / ``data_dtype`` (+ optional
        ``channel_names``); three metadata groups (``physical`` /
        ``numerical`` / ``provenance``) carrying open-content attrs.
        """
        data_np = self.data.detach().cpu().numpy()
        times_np = self.times.detach().cpu().numpy()
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data_np)
            f.create_dataset("times", data=times_np)
            f.attrs["format"] = FORMAT_NAME
            f.attrs["version"] = CURRENT_TRAJECTORY_CACHE_FORMAT.version
            f.attrs["dyn"] = self.dyn
            f.attrs["data_dtype"] = str(np.dtype(data_np.dtype).name)
            if self.channel_names is not None:
                f.attrs["channel_names"] = encode_attr(self.channel_names)
            write_group_attrs(f.create_group("physical"), self.physical)
            write_group_attrs(f.create_group("numerical"), self.numerical)
            write_group_attrs(f.create_group("provenance"), self.provenance)

    @classmethod
    def load_hdf5(cls, path: str | Path) -> TrajectoryDataset:
        """Read a dataset previously written by :meth:`save_hdf5`.

        Assumes a valid v1 cache file; run :func:`mesokit.data.validate` first
        for untrusted inputs.
        """
        with h5py.File(path, "r") as f:
            data = torch.as_tensor(f["data"][...])
            times = torch.as_tensor(f["times"][...])
            dyn = decode_attr(f.attrs["dyn"])
            channel_names_raw = decode_attr(f.attrs["channel_names"]) if "channel_names" in f.attrs else None
            channel_names = tuple(channel_names_raw) if channel_names_raw is not None else None
            physical = read_group_attrs(f.get("physical"))
            numerical = read_group_attrs(f.get("numerical"))
            provenance = read_group_attrs(f.get("provenance"))
        return cls(
            data=data,
            times=times,
            dyn=dyn,
            physical=physical,
            numerical=numerical,
            provenance=provenance,
            channel_names=channel_names,
        )

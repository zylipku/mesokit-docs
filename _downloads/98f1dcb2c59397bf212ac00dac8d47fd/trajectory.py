"""TrajectoryDataset — canonical container for trajectory data.

Holds a tensor of shape ``(n_traj, n_time, *spatial)`` plus a mesh, time grid,
and free-form metadata dict. Every model and every interpretability extractor
consumes this object — see ``docs/ARCHITECTURE.md``.

Physical / equation parameters (``eps`` for Allen-Cahn, etc.) travel as
``metadata`` so a parameter sweep is just multiple datasets sharing the same
shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import Tensor


@dataclass
class TrajectoryDataset:
    """A bundle of PDE trajectories on a fixed mesh.

    Attributes:
        data: trajectories, shape ``(n_traj, n_time, *spatial)``.
        times: time grid, shape ``(n_time,)``.
        mesh: spatial mesh; for 1D, shape ``(*spatial,)``.
        metadata: free-form physical metadata, e.g. ``{"dyn": "allen_cahn", "eps": 0.1}``.
    """

    data: Tensor
    times: Tensor
    mesh: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim < 3:
            raise ValueError(f"data must have shape (n_traj, n_time, *spatial); got {tuple(self.data.shape)}")
        n_time = self.data.shape[1]
        spatial = tuple(self.data.shape[2:])
        if tuple(self.times.shape) != (n_time,):
            raise ValueError(f"times shape {tuple(self.times.shape)} != ({n_time},)")
        if tuple(self.mesh.shape) != spatial:
            raise ValueError(f"mesh shape {tuple(self.mesh.shape)} != spatial {spatial}")

    @property
    def n_traj(self) -> int:
        return self.data.shape[0]

    @property
    def n_time(self) -> int:
        return self.data.shape[1]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[2:])

    def save_hdf5(self, path: str | Path) -> None:
        """Write the dataset to an HDF5 file. Metadata is stored on file attributes."""
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self.data.detach().cpu().numpy())
            f.create_dataset("times", data=self.times.detach().cpu().numpy())
            f.create_dataset("mesh", data=self.mesh.detach().cpu().numpy())
            for k, v in self.metadata.items():
                f.attrs[k] = v

    @classmethod
    def load_hdf5(cls, path: str | Path) -> TrajectoryDataset:
        """Read a dataset previously written by :meth:`save_hdf5`."""
        with h5py.File(path, "r") as f:
            data = torch.as_tensor(f["data"][...])
            times = torch.as_tensor(f["times"][...])
            mesh = torch.as_tensor(f["mesh"][...])
            metadata = {k: f.attrs[k] for k in f.attrs}
        return cls(data=data, times=times, mesh=mesh, metadata=metadata)

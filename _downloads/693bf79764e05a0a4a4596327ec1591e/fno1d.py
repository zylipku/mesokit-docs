"""FNO1d — Fourier Neural Operator wrapped to fit the :class:`DynamicsModel` protocol.

Delegates the heavy lifting to :class:`neuralop.models.FNO`. Implements
``rhs(u) ≈ FNO(u)`` and composes ``forward`` / ``rollout`` from the shared
explicit steppers in :mod:`mesokit.models.dynamics_base`.

Reference: Li et al., *Fourier Neural Operator for Parametric Partial
Differential Equations*, ICLR 2021.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuralop.models import FNO
from torch import Tensor, nn

from ..dynamics_base import euler_step, rk4_step, rollout
from ..registry import register

if TYPE_CHECKING:
    from mesokit.interpret import InterpretabilityArtifact


@register("fno1d")
class FNO1d(nn.Module):
    """FNO operating on 1D periodic single-channel fields, used as ``du/dt`` predictor."""

    def __init__(
        self,
        n_modes: int = 16,
        hidden_channels: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        step_method: str = "euler",
    ) -> None:
        super().__init__()
        self.step_method = step_method
        self.fno = FNO(
            n_modes=(n_modes,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def rhs(self, u: Tensor) -> Tensor:
        """Predict tendency ``du/dt(u) ≈ FNO(u)``.

        Args:
            u: ``(B, Nx)`` — single-channel 1D field.
        Returns:
            ``(B, Nx)`` — predicted ``du/dt`` at each grid point.
        """
        # neuralop expects (B, C, *spatial); add and remove the channel dim.
        return self.fno(u.unsqueeze(1)).squeeze(1)

    def forward(self, u: Tensor, dt: float) -> Tensor:
        """Advance one step of size ``dt`` using the configured stepper."""
        if self.step_method == "euler":
            return euler_step(self.rhs, u, dt)
        if self.step_method == "rk4":
            return rk4_step(self.rhs, u, dt)
        raise ValueError(f"unknown step_method: {self.step_method!r}")

    def rollout(self, u0: Tensor, n_steps: int, dt: float) -> Tensor:
        """Iterate :meth:`forward` ``n_steps`` times from ``u0``."""
        return rollout(self.forward, u0, n_steps, dt)

    def list_artifacts(self) -> list[str]:
        """Operator-family diagnostic. Currently planned, not yet implemented."""
        return ["spectrum"]

    def artifact(self, name: str) -> InterpretabilityArtifact:
        if name not in self.list_artifacts():
            raise KeyError(f"unknown artifact {name!r}; available: {self.list_artifacts()}")
        raise NotImplementedError(f"FNO1d artifact {name!r} is planned but not yet implemented; see docs/ROADMAP.md")

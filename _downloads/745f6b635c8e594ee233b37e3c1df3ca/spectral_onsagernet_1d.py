"""Spectral OnsagerNet — Onsager-structured dynamics in Fourier space.

The model parameterizes ``du/dt`` as the Onsager flow ``-(M + W) ∇V(u)``
where ``M`` is a positive-definite mobility, ``W`` an antisymmetric reactive
part, and ``V(u)`` a learnable potential
(:class:`~mesokit.models.spectral_onsagernet.potential.PolynomialPotential` by
default; the headline upstream choice is
:class:`~mesokit.models.spectral_onsagernet.potential.CoerciveAutogradPotentialV6`).

Working in 1D periodic Fourier space, both ``M`` and ``W`` reduce to per-mode
multipliers and the gradient ``∇V`` is computed by the potential directly in
Fourier space. The ``rhs`` method computes the full physical-space tendency;
the Fourier-space SBDF1 stepper implements ``forward`` and works with
``_rhs_hat``. This is the headline interpretable model: what the trained
network exposes is the potential at :attr:`potential`.

The 1D version below is the tutorial-grade rewrite of
``meso-SIR/src/models/components/spectral_onsagernet/spectral_onsagernet.py``;
the default config matches ``meso-SIR/configs/model/s_onsagernet.yaml``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..dynamics_base import euler_step, rk4_step, rollout
from ..registry import register
from .potential import PolynomialPotential, Potential

if TYPE_CHECKING:
    from mesokit.interpret import InterpretabilityArtifact


@register("spectral_onsagernet_1d")
class SpectralOnsagerNet1d(nn.Module):
    """Onsager dynamics in Fourier space on a 1D periodic mesh of ``Nx`` points."""

    def __init__(
        self,
        Nx: int = 256,
        Lx: float = 2.0 * math.pi,
        learn_M: bool = True,
        learn_W: bool = True,
        potential: Potential | None = None,
        step_method: str = "sbdf1",
    ) -> None:
        super().__init__()
        if Nx % 2 != 0:
            raise ValueError("Nx must be even (Fourier-space helpers assume even Nx).")
        self.Nx = Nx
        self.Lx = Lx
        self.step_method = step_method
        n_freq = Nx // 2 + 1

        self.M_coeff = float(learn_M)
        self.W_coeff = float(learn_W)
        self.Km_params = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.Kw_params = nn.Parameter(torch.randn(n_freq) * 0.01)

        self.potential = potential if potential is not None else PolynomialPotential(Nx=Nx)

    @property
    def Km(self) -> Tensor:
        """Mobility multipliers (real, ≥ 0): ``M_coeff * Km_params^2``."""
        return self.M_coeff * self.Km_params.square()

    @property
    def Kw(self) -> Tensor:
        """Reactive multipliers (purely imaginary): ``W_coeff * 1j * Kw_params``."""
        return self.W_coeff * (1j * self.Kw_params)

    def _rhs_hat(self, u_hat: Tensor) -> Tensor:
        """Fourier-space RHS: ``-(Km + Kw) · ∂V/∂u(u_hat)``."""
        return -(self.Km + self.Kw) * self.potential.dVdu_h(u_hat)

    def rhs(self, u: Tensor) -> Tensor:
        """Physical-space RHS: rfft → multiply by Onsager kernel → irfft."""
        u_hat = torch.fft.rfft(u, dim=-1)
        return torch.fft.irfft(self._rhs_hat(u_hat), n=u.shape[-1], dim=-1)

    def forward(self, u: Tensor, dt: float) -> Tensor:
        """Advance one step. SBDF1 / Euler / RK4 are supported via ``step_method``."""
        if self.step_method in {"sbdf1", "rk2"}:
            n = u.shape[-1]
            u_hat = torch.fft.rfft(u, dim=-1)
            u_hat_next = self._step_hat(u_hat, dt)
            return torch.fft.irfft(u_hat_next, n=n, dim=-1)
        if self.step_method == "euler":
            return euler_step(self.rhs, u, dt)
        if self.step_method == "rk4":
            return rk4_step(self.rhs, u, dt)
        raise ValueError(f"unknown step_method: {self.step_method!r}")

    def _step_hat(self, u_hat: Tensor, dt: float) -> Tensor:
        method = self.step_method
        if method == "sbdf1":
            # Implicit on the linear part of ∂V/∂u, explicit on the nonlinear remainder.
            kernel = self.Km + self.Kw
            L = self.potential.L_hat()
            N = self.potential.N_hat(u_hat)
            return (u_hat - dt * kernel * N) / (1.0 + dt * kernel * L)
        if method == "rk2":
            k1 = self._rhs_hat(u_hat)
            k2 = self._rhs_hat(u_hat + dt * k1)
            return u_hat + 0.5 * dt * (k1 + k2)
        raise ValueError(f"unknown spectral step_method: {method!r}")

    def rollout(self, u0: Tensor, n_steps: int, dt: float) -> Tensor:
        """Iterate :meth:`forward` ``n_steps`` times from ``u0``."""
        return rollout(self.forward, u0, n_steps, dt)

    def list_artifacts(self) -> list[str]:
        """Headline artifact: the learned potential landscape."""
        return ["potential"]

    def artifact(self, name: str) -> InterpretabilityArtifact:
        if name not in self.list_artifacts():
            raise KeyError(f"unknown artifact {name!r}; available: {self.list_artifacts()}")
        from .artifacts import PotentialLandscape

        return PotentialLandscape.from_model(
            self,
            model_name=getattr(self, "_mesokit_model_name", "spectral_onsagernet_1d"),
            dyn_name=getattr(self, "_mesokit_dyn_name", "unknown"),
            artifact_name=name,
            reference_dataset=getattr(self, "_mesokit_reference_dataset", None),
            max_snapshots=getattr(self, "_mesokit_artifact_max_snapshots", 5000),
        )

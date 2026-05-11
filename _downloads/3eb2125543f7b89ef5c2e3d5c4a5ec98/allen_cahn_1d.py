"""Allen-Cahn equation in 1D — Dedalus-v3 SBDF2 integrator.

Equation: ``u_t = u_xx + (u - u^3) / eps^2`` on a periodic 1D mesh.

Faithful port of ``meso-SIR/data_generation/allen_cahn.py``: same Dedalus
basis (``RealFourier`` with ``dealias=2``), same SBDF2 integrator, same
random-IC distribution (sum-of-sines, ``std``-normalized then rescaled to
``[-1, 1]``), and the same save cadence (``T_save_steps`` *iterations*
between saves — not number of save points). Defaults match
``meso-SIR/data_generation/generate_allen_cahn_data.sh`` so calling
``simulate("allen_cahn_1d")`` reproduces that dataset.

Registered as ``"allen_cahn_1d"``.
"""

from __future__ import annotations

import dedalus.public as d3
import numpy as np
import torch

from .._progress import solver_progress
from ..registry import register
from ..trajectory import TrajectoryDataset


@register("allen_cahn_1d")
def simulate_allen_cahn_1d(
    *,
    eps: float = 0.1,
    n_traj: int = 1000,
    Nx: int = 256,
    Lx: float = 2.0 * np.pi,
    T: float = 0.1,
    T_save_steps: int = 25,
    timestep: float = 1e-3 / 25,
    Nk: int = 3,
    max_k: int = 10,
    u_bound: float = 10.0,
    dealias: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> TrajectoryDataset:
    """Generate Allen-Cahn solutions on a periodic 1D mesh.

    Equation: ``u_t = u_xx + (u - u^3) / eps^2``, ``x ∈ [0, Lx]``, periodic.

    Args:
        eps: interface-width parameter.
        n_traj: number of *accepted* trajectories. Random draws repeat until
            this many pass the ``u_bound`` filter.
        Nx: spatial resolution; must be even.
        Lx: domain length.
        T: final simulated time.
        T_save_steps: save the field every ``T_save_steps`` solver iterations.
            With defaults (``T=0.1, timestep=4e-5, T_save_steps=25``) this
            yields ``1 + 100 = 101`` frames per trajectory.
        timestep: solver step size.
        Nk: number of Fourier modes selected (out of ``max_k``) for the IC.
        max_k: largest Fourier mode index considered for the IC.
        u_bound: reject trajectories whose ``|u|`` peak exceeds this.
        dealias: Dedalus de-aliasing factor.
        seed: master RNG seed; ``-1`` for non-deterministic.
        verbose: if ``True``, leave Dedalus's per-step INFO logs in place and
            don't print our own progress line. Defaults to ``False`` (quiet:
            mute Dedalus chatter, print ``m/n trajectories`` to stderr).

    Returns:
        TrajectoryDataset of shape ``(n_traj, n_frames, 1, Nx)`` with
        ``channel_names=("u",)``.
    """
    if Nx % 2 != 0:
        raise ValueError("Nx must be even (Fourier-space helpers assume even Nx).")

    coord = d3.Coordinate("x")
    dist = d3.Distributor(coord, dtype=np.float64)
    xbasis = d3.RealFourier(coord, size=Nx, bounds=(0.0, Lx), dealias=dealias)
    x = np.array(dist.local_grid(xbasis))

    rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()

    accepted: list[np.ndarray] = []
    sim_times: list[float] | None = None

    with solver_progress(dyn="allen_cahn_1d", total=n_traj, verbose=verbose) as report:
        while len(accepted) < n_traj:
            # Random IC: sum of sines (std-normalized), then rescale to [-1, 1] for AC.
            ic = _random_periodic_ic(x, Nk=Nk, max_k=max_k, rng=rng)
            ic = _rescale_to_unit_interval(ic)

            u = dist.Field(name="u", bases=xbasis)
            u["g"] = ic

            problem = d3.IVP([u])
            u_xx = d3.Differentiate(d3.Differentiate(u, coord), coord)
            problem.add_equation([d3.dt(u) - u_xx, (u - u**3) / eps**2])
            solver = problem.build_solver(d3.SBDF2)
            solver.stop_sim_time = T

            frames: list[np.ndarray] = []
            times: list[float] = []
            # change_scales(1) restores the un-dealiased grid before reading u["g"];
            # without it the dealias=2 factor leaks the 2*Nx-padded grid into the output.
            u.change_scales(1)
            frames.append(np.array(u["g"], copy=True))
            times.append(float(solver.sim_time))
            while solver.proceed:
                solver.step(timestep)
                if solver.iteration % T_save_steps == 0:
                    u.change_scales(1)
                    frames.append(np.array(u["g"], copy=True))
                    times.append(float(solver.sim_time))

            traj = np.stack(frames, axis=0)
            if not np.isfinite(traj).all() or np.abs(traj).max() > u_bound:
                continue  # rejected — try the next random IC

            if sim_times is None:
                sim_times = times
            accepted.append(traj)
            report(len(accepted))

    assert sim_times is not None
    # Insert the channel axis (single field `u`) at position 2.
    data = np.stack(accepted, axis=0).astype(np.float32)[:, :, None, :]
    return TrajectoryDataset(
        data=torch.from_numpy(data),
        times=torch.tensor(sim_times, dtype=torch.float32),
        dyn="allen_cahn_1d",
        physical={"Lx": float(Lx), "T": float(T), "eps": float(eps)},
        numerical={
            "Nx": int(Nx),
            "dx": float(Lx) / int(Nx),
            "dt": float(timestep),
            "T_save_steps": int(T_save_steps),
            "Nk": int(Nk),
            "max_k": int(max_k),
            "dealias": int(dealias),
        },
        provenance={"seed": int(seed), "u_bound": float(u_bound)},
        channel_names=("u",),
    )


def _random_periodic_ic(x: np.ndarray, *, Nk: int, max_k: int, rng: np.random.Generator) -> np.ndarray:
    """Random sum-of-sines IC, normalized to ``std=1``.

    Mirrors meso-SIR's ``InitialCondition._random_field``: select ``Nk`` modes
    from ``{1, ..., max_k}`` without replacement; draw amplitudes and phases
    for *all* ``max_k`` modes from ``Uniform(0, 1)`` and ``Uniform(0, 2π)``,
    masking unselected modes to zero frequency. The draw order
    (``selected → amp → phs``) matches meso-SIR so a shared seed produces the
    same IC sequence.

    Note: angular frequencies use ``x_len = x[-1] - x[0]`` (the discrete-grid
    span) as the period — not ``Lx``. This matches meso-SIR's quirk; for an
    ``Nx``-point periodic mesh the two differ by a factor of ``(Nx-1)/Nx``,
    enough to perturb every IC value by ~10⁻¹ if you swap them.
    """
    selected_idx = rng.choice(max_k, size=Nk, replace=False)
    mask = np.zeros(max_k, dtype=np.float64)
    mask[selected_idx] = 1.0
    x_len = x[-1] - x[0]
    angular_freq = 2.0 * np.pi * np.arange(1, max_k + 1) * mask / x_len
    amp = rng.uniform(size=max_k)
    phs = 2.0 * np.pi * rng.uniform(size=max_k)
    field = (amp[:, None] * np.sin(angular_freq[:, None] * x[None, :] + phs[:, None])).sum(axis=0)
    return field / np.std(field)


def _rescale_to_unit_interval(field: np.ndarray) -> np.ndarray:
    """Allen-Cahn post-IC step: shift and scale so the field range is ``[-1, 1]``."""
    field = field - field.min()
    field = field / field.max()
    return 2.0 * field - 1.0

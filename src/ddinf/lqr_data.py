"""The data-driven regulator: Theorem ``thm:lqr-data`` and Remark ``rmk:lqr-numerics``.

The model never appears.  The cost ``eq:lqr-cost-traj`` is a functional of the
measured signals alone, and the fundamental lemma replaces "be a trajectory" by
a condition on one record.  What remains is a quadratic problem on an affine
subspace, discretised by restricting it to finitely many admissible directions.

Those directions come from the record itself: if it is available on
``[0, T + Theta]`` then, by linearity and time invariance,

    z_i = int_0^Theta varphi_i(s) (u_bar, x_bar, y_bar)(s + .) ds

is again a trajectory for any kernel ``varphi_i``, and so is any combination
``z_g = sum_i g_i z_i``.  Taking ``varphi_i`` concentrated at ``sigma_i`` gives
the time-shifted windows used below -- trajectories exactly, with no quadrature
error.  The initial condition can only be met approximately, so it is relaxed
by a penalty ``rho`` and one minimises

    J(z_g) + rho^{-1} ||x_g(0) - x_0||_W^2
      = g'(H + rho^{-1} E) g - 2 rho^{-1} g'e + rho^{-1}||x_0||_W^2,

whose stationary points solve ``(H + rho^{-1} E) g = rho^{-1} e`` with

    H_ij = <x_i(T), G x_j(T)> + int_0^T <y_i,y_j>_Y + <R u_i, u_j>_U dt,
    E_ij = <x_i(0), x_j(0)>_W,      e_i = <x_i(0), x_0>_W.

Every entry is an inner product of measured signals.  The system is always
solvable: ``H`` and ``E`` are symmetric positive semidefinite, so any ``g`` in
the kernel of ``H + rho^{-1}E`` has ``g'Eg = ||x_g(0)||_W^2 = 0`` and hence
``g'e = <x_g(0), x_0>_W = 0``, putting ``e`` in the range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lqr_model import LqrWeights
from .moments import quadrature_weights
from .systems import LinearSystem
from .timestepping import Record


@dataclass
class DataLqrSolution:
    """The regulator recovered from the record."""

    g: np.ndarray
    record: Record  # the synthesised optimal trajectory z_g
    cost: float  # J(z_g), evaluated on measured signals only
    initial_defect: float  # ||x_g(0) - x0||_W
    rho: float
    H: np.ndarray
    E: np.ndarray
    e: np.ndarray

    @property
    def n_library(self) -> int:
        return self.g.size


def shift_library(record: Record, horizon: float, n_traj: int, *,
                  spread: float | None = None) -> list[Record]:
    """``n_traj`` time-shifted windows of length ``horizon`` from one long record.

    Each window is a trajectory of the same system with a different initial
    state, by time invariance; this is the ``varphi_i = delta_{sigma_i}`` case
    of the convolution construction, exact by definition.
    """
    window_samples = int(round(horizon / record.dt)) + 1
    if window_samples > record.n_samples:
        raise ValueError("record is shorter than the control horizon")
    last = record.n_samples - window_samples
    if spread is not None:
        last = min(last, int(round(spread / record.dt)))
    if last < n_traj - 1:
        raise ValueError(f"record too short for {n_traj} distinct shifts")
    starts = np.unique(np.round(np.linspace(0, last, n_traj)).astype(int))
    return [record.shifted(int(k), window_samples) for k in starts]


def kernel_library(record: Record, horizon: float, kernels: np.ndarray,
                   theta_len: float) -> list[Record]:
    """Library from general kernels: ``z_i(t) = int_0^Theta varphi_i(s) z(t+s) ds``.

    ``kernels`` has shape ``(N, n_s)`` sampled on ``[0, Theta]``.  Smoothing the
    shifts this way trades a little conditioning for robustness to noise.
    """
    window_samples = int(round(horizon / record.dt)) + 1
    n_s = kernels.shape[1]
    if window_samples + n_s > record.n_samples:
        raise ValueError("record too short for the requested kernels")
    s_grid = np.linspace(0.0, theta_len, n_s)
    w = quadrature_weights(s_grid) if n_s > 2 else np.full(n_s, theta_len / n_s)
    out = []
    for phi in kernels:
        coef = w * phi
        u = sum(c * record.u[:, j : j + window_samples] for j, c in enumerate(coef))
        x = sum(c * record.x[:, j : j + window_samples] for j, c in enumerate(coef))
        y = sum(c * record.y[:, j : j + window_samples] for j, c in enumerate(coef))
        out.append(Record(record.t[:window_samples] - record.t[0], u, x, y))
    return out


def assemble(library: list[Record], sys: LinearSystem, weights: LqrWeights,
             x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``H``, ``E``, ``e`` from measured signals only."""
    N = len(library)
    t = library[0].t
    w = quadrature_weights(t)

    U = np.stack([z.u for z in library])  # (N, m, T)
    Y = np.stack([z.y for z in library])  # (N, p, T)
    XT = np.stack([z.x[:, -1] for z in library])  # (N, n)
    X0 = np.stack([z.x[:, 0] for z in library])  # (N, n)

    RU = np.einsum("ab,nbt->nat", weights.R, U)
    H = (np.einsum("nat,mat,t->nm", Y, Y, w)
         + np.einsum("nat,mat,t->nm", RU, U, w)
         + XT @ weights.G @ XT.T)
    H = 0.5 * (H + H.T)

    E = X0 @ sys.MW @ X0.T
    E = 0.5 * (E + E.T)
    e = X0 @ sys.MW @ x0
    return H, E, e


def solve_data_lqr(library: list[Record], sys: LinearSystem, weights: LqrWeights,
                   x0: np.ndarray, *, rho: float = 1e-8) -> DataLqrSolution:
    """Solve ``(H + rho^{-1} E) g = rho^{-1} e`` and synthesise ``z_g``."""
    H, E, e = assemble(library, sys, weights, x0)
    Mmat = H + E / rho
    g, *_ = np.linalg.lstsq(Mmat, e / rho, rcond=None)

    t = library[0].t
    u = sum(gi * z.u for gi, z in zip(g, library))
    x = sum(gi * z.x for gi, z in zip(g, library))
    y = sum(gi * z.y for gi, z in zip(g, library))
    zg = Record(t, u, x, y)

    d = zg.x[:, 0] - x0
    return DataLqrSolution(
        g=g, record=zg, cost=float(g @ H @ g),
        initial_defect=float(np.sqrt(max(d @ sys.MW @ d, 0.0))),
        rho=rho, H=H, E=E, e=e,
    )

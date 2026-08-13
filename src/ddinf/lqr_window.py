"""Window-informative finite-dimensional LQR surrogate.

This is the shifted/convolutional trajectory implementation analysed in
``paper_wfl2/window-informativity.tex``.  It is intentionally separate from
``ddinf.lqr_data``, which discretises the synthesis-range characterization of
the main paper directly.

Given a record on ``[0, T + Theta]``, shifted windows (or convolutions of
windows) provide genuine trajectories on ``[0, T]``.  A finite library
``z_i = (u_i, x_i, y_i)`` is combined as ``z_g = sum_i g_i z_i`` and the
initial condition is imposed with the penalty

    J(z_g) + rho^{-1} ||x_g(0) - x0||_W^2.

The stationary equation is

    (H + rho^{-1} E) g = rho^{-1} e.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lqr_model import LqrWeights
from .moments import quadrature_weights, trapezoid_weights
from .systems import LinearSystem
from .timestepping import Record


@dataclass
class WindowLqrSolution:
    """Regulator reconstructed in a finite window-trajectory library."""

    g: np.ndarray
    record: Record
    cost: float
    initial_defect: float
    rho: float
    H: np.ndarray
    E: np.ndarray
    e: np.ndarray

    @property
    def n_library(self) -> int:
        return self.g.size


def shift_library(record: Record, horizon: float, n_traj: int, *,
                  spread: float | None = None) -> list[Record]:
    """Return ``n_traj`` shifted windows of the prescribed horizon."""
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
    """Build trajectories ``z_i(t)=int phi_i(s) z_bar(t+s) ds``."""
    window_samples = int(round(horizon / record.dt)) + 1
    n_s = kernels.shape[1]
    if window_samples + n_s > record.n_samples:
        raise ValueError("record too short for the requested kernels")
    s_grid = np.linspace(0.0, theta_len, n_s)
    w = quadrature_weights(s_grid) if n_s > 2 else np.full(n_s, theta_len / n_s)
    out = []
    for phi in kernels:
        coef = w * phi
        u = sum(c * record.u[:, j:j + window_samples] for j, c in enumerate(coef))
        x = sum(c * record.x[:, j:j + window_samples] for j, c in enumerate(coef))
        y = sum(c * record.y[:, j:j + window_samples] for j, c in enumerate(coef))
        out.append(Record(record.t[:window_samples] - record.t[0], u, x, y))
    return out


def assemble_window_lqr(library: list[Record], sys: LinearSystem,
                        weights: LqrWeights, x0: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the measured-signal matrices ``H``, ``E`` and ``e``."""
    t = library[0].t
    w_y = quadrature_weights(t)
    w_u = trapezoid_weights(t)

    U = np.stack([z.u for z in library])
    Y = np.stack([z.y for z in library])
    XT = np.stack([z.x[:, -1] for z in library])
    X0 = np.stack([z.x[:, 0] for z in library])

    RU = np.einsum("ab,nbt->nat", weights.R, U)
    H = (np.einsum("nat,mat,t->nm", Y, Y, w_y)
         + np.einsum("nat,mat,t->nm", RU, U, w_u)
         + XT @ weights.G @ XT.T)
    H = 0.5 * (H + H.T)

    E = X0 @ sys.MW @ X0.T
    E = 0.5 * (E + E.T)
    e = X0 @ sys.MW @ x0
    return H, E, e


def solve_window_lqr(library: list[Record], sys: LinearSystem,
                     weights: LqrWeights, x0: np.ndarray, *, rho: float = 1e-8
                     ) -> WindowLqrSolution:
    """Solve the penalized LQR problem in a window-trajectory library."""
    H, E, e = assemble_window_lqr(library, sys, weights, x0)
    g, *_ = np.linalg.lstsq(H + E / rho, e / rho, rcond=None)

    t = library[0].t
    u = sum(gi * z.u for gi, z in zip(g, library))
    x = sum(gi * z.x for gi, z in zip(g, library))
    y = sum(gi * z.y for gi, z in zip(g, library))
    zg = Record(t, u, x, y)

    d = zg.x[:, 0] - x0
    return WindowLqrSolution(
        g=g,
        record=zg,
        cost=float(g @ H @ g),
        initial_defect=float(np.sqrt(max(d @ sys.MW @ d, 0.0))),
        rho=rho,
        H=H,
        E=E,
        e=e,
    )


# Backwards-compatible names for users of the original implementation.
DataLqrSolution = WindowLqrSolution
assemble = assemble_window_lqr
solve_data_lqr = solve_window_lqr

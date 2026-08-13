"""Theta-Euler time stepping and the measured record.

For ``x' = A x + B u`` the scheme is

    (I - theta dt A) x^{k+1} = (I + (1-theta) dt A) x^k
                               + dt (theta B u^{k+1} + (1-theta) B u^k),

so ``theta = 1`` is backward Euler, ``theta = 0`` forward Euler and
``theta = 1/2`` Crank--Nicolson.  ``theta = 1`` is the default because the
semi-discrete heat operator is stiff (``|lambda_max| ~ nu h^{-2}``, so forward
Euler would need ``dt < h^2/(2 nu)``), but every experiment in this project
passes ``theta = 1/2``: only Crank--Nicolson is second-order accurate, which is
what the moment identities and the LQR cost are scored at.

Note that for ``theta = 1/2`` the input enters a step only through the two-point
average ``(u_k + u_{k+1})/2``.  The odd--even component of a sampled input is
therefore invisible to the state, which is why the LQR cost has to weight its
control term with :func:`ddinf.moments.trapezoid_weights`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .systems import LinearSystem


@dataclass
class Record:
    """A sampled input--state--output trajectory ``(u, x, y)`` on a time grid."""

    t: np.ndarray  # (T,)
    u: np.ndarray  # (m, T)
    x: np.ndarray  # (n, T)
    y: np.ndarray  # (p, T)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def n_samples(self) -> int:
        return self.t.size

    @property
    def horizon(self) -> float:
        return float(self.t[-1] - self.t[0])

    def window(self, t0: float, t1: float, *, rebase: bool = True) -> "Record":
        """Sub-record on ``[t0, t1]``; ``rebase`` shifts its clock back to 0."""
        idx = np.where((self.t >= t0 - 1e-12) & (self.t <= t1 + 1e-12))[0]
        t = self.t[idx]
        return Record(t - t[0] if rebase else t,
                      self.u[:, idx], self.x[:, idx], self.y[:, idx])

    def shifted(self, k: int, n_samples: int) -> "Record":
        """The length-``n_samples`` window starting ``k`` samples in, clock at 0.

        By time invariance this is again a trajectory of the same system, with
        initial state ``x(t_k)``; this is what supplies the library directions
        of Remark ``rmk:lqr-numerics``.
        """
        sl = slice(k, k + n_samples)
        return Record(self.t[sl] - self.t[k], self.u[:, sl], self.x[:, sl], self.y[:, sl])


def simulate(sys: LinearSystem, u: Callable[[np.ndarray], np.ndarray],
             t: np.ndarray, x0: np.ndarray, *, theta: float = 1.0) -> Record:
    """Integrate ``x' = Ax + Bu``, ``y = Cx`` on the grid ``t`` by the theta scheme."""
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("theta stepping expects a uniform time grid")

    U = np.atleast_2d(u(t))
    if U.shape[0] != sys.m:
        raise ValueError(f"input has {U.shape[0]} components, system expects {sys.m}")

    n = sys.n
    lu = lu_factor(np.eye(n) - theta * dt * sys.A)
    # One-step propagator and forcing map, precomputed: the grid is uniform, so
    # the recursion costs a single mat-vec per step instead of a linear solve.
    P = lu_solve(lu, np.eye(n) + (1.0 - theta) * dt * sys.A)
    Q = lu_solve(lu, dt * sys.B)
    F = Q @ (theta * U[:, 1:] + (1.0 - theta) * U[:, :-1])  # (n, T-1)

    X = np.empty((n, t.size))
    X[:, 0] = x0
    for k in range(t.size - 1):
        X[:, k + 1] = P @ X[:, k] + F[:, k]

    return Record(t, U, X, sys.C @ X)


def uniform_grid(horizon: float, dt: float) -> np.ndarray:
    """Grid ``0, dt, ..., horizon`` with ``horizon`` an exact multiple of ``dt``."""
    n = int(round(horizon / dt))
    if abs(n * dt - horizon) > 1e-9 * max(1.0, horizon):
        raise ValueError("horizon must be an integer multiple of dt")
    return np.linspace(0.0, horizon, n + 1)

"""Model-based LQR: the reference the data-driven regulator is measured against.

For the cost of ``eq:lqr-cost``,

    J(u; x0) = <x(T), G x(T)> + int_0^T ||C x||_Y^2 + <u, R u>_U dt,

Theorem ``thm:lqr-riccati`` gives the optimum through the Riccati operator.  In
the semi-discrete setting that is the differential Riccati equation

    -P' = A'P + PA - P B R^{-1} B' P + C'C,      P(T) = G,

which is solved here *in closed form* rather than by integration: with the
Hamiltonian

    H = [[A, -B R^{-1} B'], [-C'C, -A']],

the Riccati flow is the Riccati transform of ``exp(-H s)``,

    P(T-s) = (Phi21 + Phi22 G)(Phi11 + Phi12 G)^{-1},   Phi = exp(-H s),

which follows from propagating the Hamiltonian two-point boundary value problem
backwards from ``p(T) = G x(T)``.  Advancing ``s`` by one step at a time costs a
single matrix product, and :func:`riccati_ivp` re-derives the same object with
an adaptive integrator as an independent check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, lu_factor, lu_solve

from .systems import LinearSystem
from .timestepping import Record


@dataclass
class LqrWeights:
    """The cost weights ``(G, R)``; the state penalty is always ``C'C``."""

    G: np.ndarray
    R: np.ndarray

    @staticmethod
    def make(sys: LinearSystem, *, terminal: float = 1.0, control: float = 1.0
             ) -> "LqrWeights":
        """``G = terminal * M_X`` (i.e. ``terminal ||x(T)||_X^2``) and ``R = control I``."""
        return LqrWeights(G=terminal * sys.MX, R=control * np.eye(sys.m))


@dataclass
class RiccatiSolution:
    """``P(t)`` on a time grid, with the feedback it induces."""

    t: np.ndarray
    P: np.ndarray  # (T, n, n)
    sys: LinearSystem
    weights: LqrWeights

    def gain(self, k: int) -> np.ndarray:
        """``F(t_k) = -R^{-1} B' P(t_k)``."""
        return -np.linalg.solve(self.weights.R, self.sys.B.T @ self.P[k])

    def optimal_cost(self, x0: np.ndarray) -> float:
        """``J* = <x0, P(0) x0>``."""
        return float(x0 @ self.P[0] @ x0)

    def closed_loop(self, x0: np.ndarray, *, theta: float = 0.5) -> Record:
        """Simulate ``x' = (A + B F(t)) x`` and return the optimal record.

        The feedback is time varying, so the propagator is rebuilt each step;
        the same theta scheme as :func:`ddinf.timestepping.simulate` is used.
        """
        t, sys = self.t, self.sys
        dt = float(t[1] - t[0])
        n = sys.n
        X = np.empty((n, t.size))
        U = np.empty((sys.m, t.size))
        X[:, 0] = x0
        for k in range(t.size - 1):
            Ak = sys.A + sys.B @ self.gain(k)
            Ak1 = sys.A + sys.B @ self.gain(k + 1)
            lhs = np.eye(n) - theta * dt * Ak1
            rhs = (np.eye(n) + (1.0 - theta) * dt * Ak) @ X[:, k]
            X[:, k + 1] = lu_solve(lu_factor(lhs), rhs)
        for k in range(t.size):
            U[:, k] = self.gain(k) @ X[:, k]
        return Record(t, U, X, sys.C @ X)


def riccati_hamiltonian(sys: LinearSystem, weights: LqrWeights,
                        t: np.ndarray) -> RiccatiSolution:
    """Closed-form ``P(t)`` by propagating ``exp(-H s)`` along the grid."""
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    n = sys.n
    Rinv = np.linalg.inv(weights.R)
    H = np.block([
        [sys.A, -sys.B @ Rinv @ sys.B.T],
        [-sys.C.T @ sys.C, -sys.A.T],
    ])
    step = expm(-H * dt)

    P = np.empty((t.size, n, n))
    Phi = np.eye(2 * n)
    for j in range(t.size):  # j counts backwards from T: s = j * dt
        top = Phi[:n, :n] + Phi[:n, n:] @ weights.G
        bot = Phi[n:, :n] + Phi[n:, n:] @ weights.G
        Pk = np.linalg.solve(top.T, bot.T).T
        P[t.size - 1 - j] = 0.5 * (Pk + Pk.T)
        Phi = Phi @ step
    return RiccatiSolution(t, P, sys, weights)


def riccati_ivp(sys: LinearSystem, weights: LqrWeights, t: np.ndarray, *,
                rtol: float = 1e-11, atol: float = 1e-13) -> RiccatiSolution:
    """``P(t)`` by backward integration of the DRE -- an independent check."""
    t = np.asarray(t, dtype=float)
    n = sys.n
    Rinv = np.linalg.inv(weights.R)
    Q = sys.C.T @ sys.C
    S = sys.B @ Rinv @ sys.B.T

    def rhs(_s: float, p: np.ndarray) -> np.ndarray:
        # integrating in s = T - t, so dP/ds = +(A'P + PA - PSP + Q)
        Pm = p.reshape(n, n)
        d = sys.A.T @ Pm + Pm @ sys.A - Pm @ S @ Pm + Q
        return d.ravel()

    s_eval = t[-1] - t[::-1]
    sol = solve_ivp(rhs, (0.0, float(s_eval[-1])), weights.G.ravel(),
                    t_eval=s_eval, rtol=rtol, atol=atol, method="Radau")
    P = sol.y.T.reshape(-1, n, n)[::-1]
    return RiccatiSolution(t, 0.5 * (P + np.swapaxes(P, 1, 2)), sys, weights)


def trajectory_cost(rec: Record, sys: LinearSystem, weights: LqrWeights) -> float:
    """``J`` evaluated on a record -- the functional ``eq:lqr-cost-traj``.

    Only measured signals enter: the terminal state, the output and the input.
    """
    w = np.full(rec.n_samples, rec.dt)
    if rec.n_samples % 2 == 1:
        w = np.ones(rec.n_samples)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= rec.dt / 3.0
    else:
        w[0] = w[-1] = rec.dt / 2.0
    running = (rec.y * rec.y).sum(axis=0) + (rec.u * (weights.R @ rec.u)).sum(axis=0)
    terminal = float(rec.x[:, -1] @ weights.G @ rec.x[:, -1])
    return terminal + float(w @ running)

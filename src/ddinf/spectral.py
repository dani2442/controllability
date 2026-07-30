"""Closed-form modal references for the heat equation examples.

These are the ground truth the FEM + Euler discretisation is measured against.
For a diagonalisable generator ``A phi_n = lambda_n phi_n``, the mild solution

    x(t) = S(t) x_0 + int_0^t S(t-s) B u(s) ds

is, mode by mode, an explicit sum of exponentials whenever the input is one --
which is exactly the class :class:`ddinf.signals.ExpSum`.

Boundary control and the lift
-----------------------------
Written directly in the eigenbasis, boundary control has slowly converging
modal coefficients: for Dirichlet control ``b_n = sqrt(2) n pi`` grows, the
solution does not vanish at ``xi = 0``, and its sine series converges only like
``1/n``.  The reference is therefore evaluated in *lifted* form

    x(t, xi) = u(t) D(xi) + sum_n v_n(t) phi_n(xi),
    v_n' = lambda_n v_n + g_n u - d_n u',   d_n = <D, phi_n>, g_n = <D'', phi_n>,

with ``D`` a fixed function carrying the inhomogeneous boundary condition.
The remainder ``v`` satisfies homogeneous boundary conditions, its coefficients
decay like ``n^{-3}``, and a few hundred modes give a reference accurate to
``1e-8``.  The two forms agree analytically: integrating ``int e^{lam(t-s)}u'``
by parts turns ``-d_n u'`` into ``b_n u`` with ``b_n = -lambda_n d_n``.

Modal data (see Pritchard--Salamon Ex. 4.6 and ``sections/lqr.tex``):

===========================  =============  ========================  =========================
example                      ``lambda_n``   ``phi_n``                 ``b_n``
===========================  =============  ========================  =========================
Dirichlet control at 0       ``-n^2pi^2``   ``sqrt2 sin(n pi xi)``    ``sqrt2 n pi``
Dirichlet control both ends  ``-n^2pi^2``   ``sqrt2 sin(n pi xi)``    ``sqrt2 n pi (1-(-1)^n)``
Neumann control at 0         ``-n^2pi^2``   ``sqrt2 cos(n pi xi)``    ``-phi_n(0)``
===========================  =============  ========================  =========================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .signals import ExpSum


def _simpson_weights(n_quad: int) -> np.ndarray:
    if n_quad % 2 == 0:
        raise ValueError("Simpson quadrature needs an odd number of points")
    w = np.ones(n_quad)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (1.0 / (n_quad - 1)) / 3.0


@dataclass
class ModalSystem:
    """Diagonal realisation of an example, with exact mild solutions."""

    name: str
    lam: np.ndarray  # (N,)
    b: np.ndarray  # (N, m) modal control coefficients
    phi: Callable[[np.ndarray], np.ndarray]  # xi -> (N, len(xi))
    lift: Callable[[np.ndarray], np.ndarray] | None = None  # D(xi)
    d: np.ndarray | None = None  # <D, phi_n>
    g: np.ndarray | None = None  # <D'', phi_n>

    @property
    def n_modes(self) -> int:
        return self.lam.shape[0]

    def modes_of(self, f: Callable[[np.ndarray], np.ndarray],
                 n_quad: int = 20001) -> np.ndarray:
        """Modal coefficients ``<f, phi_n>``, all modes in one Simpson pass."""
        xi = np.linspace(0.0, 1.0, n_quad)
        return self.phi(xi) @ (_simpson_weights(n_quad) * np.asarray(f(xi), dtype=float))

    def convolve(self, t: np.ndarray, gain: np.ndarray, u: ExpSum,
                 free_exp: np.ndarray | None = None) -> np.ndarray:
        """``int_0^t e^{lam(t-s)} gain_n u(s) ds`` for every mode, shape ``(N, T)``.

        Uses ``int_0^t e^{lam(t-s)} e^{s_k s} ds = (e^{s_k t}-e^{lam t})/(s_k-lam)``.
        Summing over the exponents first turns the whole convolution into one
        ``(N,K) @ (K,T)`` product plus a rank-one correction, so the expensive
        ``e^{lam t}`` array is formed once; pass it in as ``free_exp`` to reuse
        it across calls.  The confluent case ``s_k = lam_n`` gets ``t e^{lam t}``.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        lam = self.lam[:, None]
        E_lam = np.exp(lam * t) if free_exp is None else free_exp  # (N, T)
        E_s = np.exp(np.outer(u.exponents, t))  # (K, T)

        gain2 = gain if gain.ndim == 2 else gain[:, None]  # (N, m)
        modal_gain = gain2 @ u.coeffs.T  # (N, K)

        denom = u.exponents[None, :] - lam  # (N, K)
        confluent = np.abs(denom) < 1e-14
        safe = np.where(confluent, 1.0, denom)
        coef = modal_gain / safe  # (N, K)
        coef[confluent] = 0.0

        out = coef @ E_s - coef.sum(axis=1)[:, None] * E_lam
        if np.any(confluent):
            rows, cols = np.nonzero(confluent)
            out[rows] += (modal_gain[rows, cols][:, None]
                          * t[None, :] * E_lam[rows])
        return out

    def mild_solution_modes(self, t: np.ndarray, x0_modes: np.ndarray,
                            u: ExpSum) -> np.ndarray:
        """Modal coefficients of ``S(t)x_0 + int S(t-s)Bu(s)ds``, shape ``(N, T)``.

        Direct (unlifted) form -- correct, but for boundary control the series
        it belongs to converges slowly; prefer :meth:`solution_on_grid`.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        E_lam = np.exp(self.lam[:, None] * t)
        return E_lam * x0_modes[:, None] + np.real(self.convolve(t, self.b, u, E_lam))

    def solution_on_grid(self, t: np.ndarray, x0: Callable, u: ExpSum,
                         xi: np.ndarray) -> np.ndarray:
        """Physical solution values ``x(t, xi)``, shape ``(len(xi), T)``.

        Evaluated in lifted form when the configuration has a boundary lift.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        xi = np.asarray(xi, dtype=float)
        u_vals = np.atleast_2d(u(t))[0]

        if self.lift is None:
            modes = self.mild_solution_modes(t, self.modes_of(x0), u)
            return self.phi(xi).T @ modes

        x0_modes = self.modes_of(x0)
        u0 = float(np.atleast_2d(u(np.array([0.0])))[0, 0])
        v0_modes = x0_modes - u0 * self.d
        E_lam = np.exp(self.lam[:, None] * t)
        v = (E_lam * v0_modes[:, None]
             + np.real(self.convolve(t, self.g, u, E_lam))
             - np.real(self.convolve(t, self.d, u.derivative(), E_lam)))
        return self.lift(xi)[:, None] * u_vals[None, :] + self.phi(xi).T @ v

    def uncontrollable_modes(self, tol: float = 1e-12) -> np.ndarray:
        """Indices ``n`` with ``b_n = 0`` -- the Fattorini--Hautus obstructions."""
        return np.where(np.linalg.norm(self.b, axis=1) <= tol)[0]


def heat_modal(kind: str = "dirichlet", n_modes: int = 400, nu: float = 1.0) -> ModalSystem:
    """Closed-form modal data for the three heat-equation configurations.

    ``nu`` is the diffusivity of ``x_t = nu x_xx``; it scales the eigenvalues,
    and with them the control coefficients ``b_n = -lambda_n <D, phi_n>``.
    """
    if kind in ("dirichlet", "dirichlet_sym"):
        n = np.arange(1, n_modes + 1)
        lam = -nu * ((n * np.pi) ** 2)

        def phi(xi: np.ndarray, n=n) -> np.ndarray:
            return np.sqrt(2.0) * np.sin(np.outer(n * np.pi, np.asarray(xi, dtype=float)))

        if kind == "dirichlet":
            # D u = (1-xi) u,  <D, phi_n> = sqrt(2)/(n pi),  D'' = 0
            lift = lambda xi: 1.0 - np.asarray(xi, dtype=float)  # noqa: E731
            d = np.sqrt(2.0) / (n * np.pi)
        else:
            # D u = u (both ends),  <1, phi_n> = sqrt(2)(1-(-1)^n)/(n pi),  D'' = 0
            lift = lambda xi: np.ones_like(np.asarray(xi, dtype=float))  # noqa: E731
            d = np.sqrt(2.0) * (1.0 - (-1.0) ** n) / (n * np.pi)
        b = (-lam * d)[:, None]
        return ModalSystem(f"heat-{kind}", lam, b, phi, lift=lift, d=d, g=np.zeros_like(d))

    if kind == "neumann":
        n = np.arange(0, n_modes)
        lam = -nu * ((n * np.pi) ** 2)

        def phi(xi: np.ndarray, n=n) -> np.ndarray:
            xi = np.asarray(xi, dtype=float)
            out = np.sqrt(2.0) * np.cos(np.outer(n * np.pi, xi))
            out[0] = np.ones_like(xi)  # phi_0 == 1
            return out

        b = -nu * phi(np.array([0.0]))[:, 0][:, None]  # b_n = -nu phi_n(0)
        # D(xi) = -(1-xi)^2/2 has D'(0) = 1, D'(1) = 0, D'' = -1
        lift = lambda xi: -0.5 * (1.0 - np.asarray(xi, dtype=float)) ** 2  # noqa: E731
        d = np.array([-1.0 / 6.0] + [np.sqrt(2.0) * (-1.0) / (k * np.pi) ** 2
                                     for k in range(1, n_modes)])
        g = np.zeros(n_modes)
        g[0] = -nu  # <nu D'', phi_0> = -nu, and <nu D'', phi_n> = 0 for n >= 1
        return ModalSystem("heat-neumann", lam, b, phi, lift=lift, d=d, g=g)

    raise ValueError(f"unknown heat configuration {kind!r}")

"""P1 finite elements on the interval (0, 1).

Everything is dense: the meshes used here have at most a few hundred nodes, and
the downstream algorithms (generalised eigenproblems, matrix exponentials,
dense least squares) want dense matrices anyway.

Conventions
-----------
Nodes are ``xi_0 = 0 < xi_1 < ... < xi_N = 1`` and the basis is the usual hat
functions ``psi_j``.  A finite element function is identified with its vector of
nodal values, so that the ``L^2(0,1)`` inner product of two functions is
``a @ M @ b`` with ``M`` the mass matrix, and the ``H^1`` inner product is
``a @ (M + K) @ b``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Mesh1D:
    """Uniform P1 mesh on ``(0, 1)`` with ``n_elems`` elements."""

    n_elems: int

    @property
    def h(self) -> float:
        return 1.0 / self.n_elems

    @property
    def n_nodes(self) -> int:
        return self.n_elems + 1

    @property
    def nodes(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_nodes)


def mass_matrix(mesh: Mesh1D, *, lumped: bool = False) -> np.ndarray:
    """P1 mass matrix ``M_ij = \\int psi_i psi_j``.

    With ``lumped=True`` the row-sum lumped (diagonal) variant is returned.
    Lumping is what makes Dirichlet *control* enter the semi-discrete system
    without a ``u_dot`` term: the coupling block ``M[interior, boundary]``
    vanishes identically, so the discrete system is exactly ``x' = A x + B u``.
    """
    h = mesh.h
    n = mesh.n_nodes
    M = np.zeros((n, n))
    elem = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    for e in range(mesh.n_elems):
        M[e : e + 2, e : e + 2] += elem
    if lumped:
        M = np.diag(M.sum(axis=1))
    return M


def stiffness_matrix(mesh: Mesh1D) -> np.ndarray:
    """P1 stiffness matrix ``K_ij = \\int psi_i' psi_j'``."""
    h = mesh.h
    n = mesh.n_nodes
    K = np.zeros((n, n))
    elem = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(mesh.n_elems):
        K[e : e + 2, e : e + 2] += elem
    return K


def interpolate(mesh: Mesh1D, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Nodal interpolant of ``f`` (second-order accurate in ``L^2``)."""
    return np.asarray(f(mesh.nodes), dtype=float)


def point_evaluation(mesh: Mesh1D, xi0: float) -> np.ndarray:
    """Row weights for evaluating a P1 function at ``xi0``.

    If ``x`` contains the nodal values of a finite-element function, then
    ``point_evaluation(mesh, xi0) @ x`` is its exact P1 interpolant at ``xi0``.
    """
    if not np.isfinite(xi0) or not 0.0 <= xi0 <= 1.0:
        raise ValueError("evaluation point must belong to [0, 1]")

    weights = np.zeros(mesh.n_nodes)
    if xi0 == 1.0:
        weights[-1] = 1.0
        return weights

    element = min(int(np.floor(xi0 / mesh.h)), mesh.n_elems - 1)
    local = (xi0 - element * mesh.h) / mesh.h
    weights[element] = 1.0 - local
    weights[element + 1] = local
    return weights


def bump(xi0: float = 0.6, width: float = 0.25) -> Callable[[np.ndarray], np.ndarray]:
    """A ``C^infty`` mollifier of unit mass supported in ``(xi0-width, xi0+width)``.

    Used as the kernel ``c`` of a smooth distributed observation (and, because
    it is localized near ``xi0``, as a mollified point measurement).
    ``y(t) = \\int_0^1 c(xi) x(t,xi) dxi \\approx x(t, xi0)``.  Smoothness and
    compact support make the modal coefficients ``c_n = <c, phi_n>`` decay
    faster than any power of ``n``, so ``sum_n |c_n| < infty`` and the pair
    ``(B, C)`` is admissible in the Pritchard--Salamon scale even when ``B`` is
    the (unbounded) Dirichlet control operator.

    The support is kept away from ``xi = 0`` so that the observation of the
    boundary lift vanishes and the measured output carries no feedthrough term.
    """
    if not (0.0 < xi0 - width and xi0 + width < 1.0):
        raise ValueError("bump support must lie strictly inside (0, 1)")

    def c(xi: np.ndarray) -> np.ndarray:
        s = (np.asarray(xi, dtype=float) - xi0) / width
        out = np.zeros_like(s)
        inside = np.abs(s) < 1.0
        out[inside] = np.exp(1.0 - 1.0 / (1.0 - s[inside] ** 2))
        # normalise to unit mass by high-order quadrature on the support
        return out / _bump_mass(xi0, width)

    return c


def _bump_mass(xi0: float, width: float) -> float:
    """``\\int`` of the unnormalised bump, by fine Simpson quadrature."""
    n = 4001
    s = np.linspace(-1.0, 1.0, n)
    vals = np.zeros_like(s)
    inside = np.abs(s) < 1.0
    vals[inside] = np.exp(1.0 - 1.0 / (1.0 - s[inside] ** 2))
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return width * (2.0 / (n - 1)) / 3.0 * float(w @ vals)


def quad_inner(f: Callable, g: Callable, n: int = 20001) -> float:
    """``\\int_0^1 f g`` by fine Simpson quadrature (for closed-form references)."""
    xi = np.linspace(0.0, 1.0, n)
    vals = np.asarray(f(xi), dtype=float) * np.asarray(g(xi), dtype=float)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((1.0 / (n - 1)) / 3.0 * (w @ vals))

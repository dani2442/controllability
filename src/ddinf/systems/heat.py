"""The one-dimensional heat equation, in the three configurations of the paper.

``dirichlet``
    ``x_t = x_xx``, ``x(t,0) = u(t)``, ``x(t,1) = 0`` -- an auxiliary variant
    used for the symmetric uncontrollable comparison.  The control operator is unbounded on
    ``X = L^2(0,1)``; discretely ``||B_h||_{L(U,X)}`` grows like ``h^{-3/2}``,
    the resolution of a ``delta'`` at the boundary.

``dirichlet_sym``
    the same equation with ``x(t,0) = x(t,1) = u(t)``.  A single control cannot
    reach the modes that are antisymmetric about ``xi = 1/2``, so
    ``b_n = 0`` for every even ``n``: the reference *uncontrollable* system,
    with the obstruction available in closed form.

``neumann``
    ``x_xi(t,0) = u(t)``, ``x_xi(t,1) = 0`` -- verbatim
    Pritchard--Salamon Example 4.6 and the introduction's Example
    ``ex:intro-pde``.  The control is a *natural* boundary condition, so no
    lifting is needed.  The homogeneous generator has the constant zero mode,
    followed by ``lambda_n = -nu n^2 pi^2`` and eigenfunctions
    ``sqrt2 cos(n pi xi)``, ``n >= 1``.  Discretely ``||B_h||`` grows only
    like ``h^{-1/2}``.

Dirichlet control and mass lumping
----------------------------------
Writing ``x_h = X_I psi_I + u psi_0`` and testing against interior hat
functions gives ``M_II X_I' + M_I0 u' + K_II X_I + K_I0 u = 0``.  The ``u'``
term would take the semi-discrete system out of the form ``x' = A x + B u``
required by the fundamental lemma.  Row-sum mass lumping makes ``M`` diagonal,
so ``M_I0 = 0`` and the term disappears *exactly* -- no approximation of the
dynamics beyond the lumping itself, which is still second-order accurate.
The Neumann configuration needs no lifting and uses the consistent mass matrix.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .fem import (Mesh1D, bump, interpolate, mass_matrix, point_evaluation,
                  stiffness_matrix)
from .base import LinearSystem


def heat_system(kind: str = "neumann", *, n_elems: int = 64, nu: float = 1.0,
                obs_center: float = 0.6, obs_width: float = 0.25,
                obs: Callable[[np.ndarray], np.ndarray] | None = None,
                observation: Literal["point", "distributed"] | None = None,
                ) -> LinearSystem:
    """Semi-discrete heat equation ``x_t = nu x_xx`` as a :class:`LinearSystem`.

    By default, ``y(t) = x(t, obs_center)`` is evaluated exactly in the P1
    space.  This functional is unbounded on ``X=L^2(0,1)`` but bounded on the
    finer space ``W=H^1(0,1)``, matching Example ``ex:intro-pde``.  Set
    ``observation="distributed"`` to use a smooth unit-mass bump instead.  For
    backwards compatibility, passing a custom kernel through ``obs`` also
    selects the distributed observation when ``observation`` is omitted.

    The diffusivity ``nu`` rescales time (``lambda_n = -nu n^2 pi^2``) and
    is what makes the example numerically legible: the record can only be
    informative for the modes it excites, so the input band has to reach
    ``|lambda_max| ~ nu h^{-2}``, and the pairing ``<eta, x(t)>`` can only
    resolve a mode while ``e^{lambda t}`` stays above round-off.  Both are
    comfortable for moderate ``nu``; both fail for ``nu = 1`` on a fine mesh,
    which is the finite-precision face of the compactness of the Gramian.
    """
    mesh = Mesh1D(n_elems)
    K0 = stiffness_matrix(mesh)  # unscaled: defines the H^1 structure of W
    K = nu * K0
    obs_kind = observation or ("distributed" if obs is not None else "point")
    if obs_kind not in ("point", "distributed"):
        raise ValueError(f"unknown heat observation {obs_kind!r}")
    if obs_kind == "point" and obs is not None:
        raise ValueError("a custom kernel requires observation='distributed'")

    if obs_kind == "point":
        obs_weights = point_evaluation(mesh, obs_center)
        c_fun = None
        c_nodal = None
    else:
        c_fun = obs if obs is not None else bump(obs_center, obs_width)
        c_nodal = interpolate(mesh, c_fun)
        obs_weights = None

    if kind in ("dirichlet", "dirichlet_sym"):
        M = mass_matrix(mesh, lumped=True)
        free = np.arange(1, mesh.n_nodes - 1)  # interior nodes
        M_II = M[np.ix_(free, free)]
        K_II = K[np.ix_(free, free)]
        Minv = np.diag(1.0 / np.diag(M_II))
        A = -Minv @ K_II
        if kind == "dirichlet":
            load = K[np.ix_(free, [0])]
        else:
            load = K[np.ix_(free, [0])] + K[np.ix_(free, [mesh.n_nodes - 1])]
        B = -Minv @ load
        MX = M_II
        if obs_kind == "point":
            controlled = [0] if kind == "dirichlet" else [0, mesh.n_nodes - 1]
            if np.any(np.abs(obs_weights[controlled]) > 1e-14):
                raise ValueError(
                    "point observation must not share a finite element with a "
                    "controlled Dirichlet boundary"
                )
            C = obs_weights[free][None, :]
        else:
            C = (M_II @ c_nodal[free])[None, :]

        def expand(X: np.ndarray, u: np.ndarray | float = 0.0) -> np.ndarray:
            """Interior dofs -> full nodal values, including the boundary lift."""
            full = np.zeros((mesh.n_nodes,) + np.shape(X)[1:])
            full[free] = X
            full[0] = u
            if kind == "dirichlet_sym":
                full[-1] = u
            return full

    elif kind == "neumann":
        # Neumann control at xi = 0, homogeneous Neumann data at xi = 1.
        # Every nodal value is free and no lifting is required.  The weak
        # boundary term is -nu*u*v(0), hence B = -nu*M^{-1}*e_0.
        M = mass_matrix(mesh, lumped=False)
        free = np.arange(mesh.n_nodes)
        Minv = np.linalg.inv(M)
        A = -Minv @ K
        e0 = np.zeros((free.size, 1))
        e0[0, 0] = 1.0
        B = -nu * (Minv @ e0)
        MX = M
        if obs_kind == "point":
            C = obs_weights[None, :]
        else:
            C = (M @ c_nodal)[None, :]

        def expand(X: np.ndarray, u: np.ndarray | float = 0.0) -> np.ndarray:
            return np.asarray(X)

    else:
        raise ValueError(f"unknown heat configuration {kind!r}")

    MW = MX + K0[np.ix_(free, free)]  # discrete H^1 inner product (independent of nu)

    return LinearSystem(
        name=f"heat-{kind}(n={n_elems}, nu={nu:g})",
        A=A, B=B, C=C, MX=MX, MW=MW,
        meta={
            "family": "heat",
            "kind": kind,
            "nu": nu,
            "mesh": mesh,
            "free": free,
            "expand": expand,
            "observation": obs_kind,
            "obs_center": obs_center,
            "obs_weights": obs_weights,
            "obs_kernel": c_fun,
            "obs_nodal": c_nodal,
            "modal_kind": "neumann" if kind == "neumann" else "dirichlet",
        },
    )


def fem_eigenvalues(sys: LinearSystem, k: int | None = None) -> np.ndarray:
    """Eigenvalues of the semi-discrete generator, sorted by decreasing real part.

    These are the exact obstruction candidates for the *discretized* system, and
    approximate ``-n^2 pi^2`` to second order in ``h``.
    """
    lam = np.linalg.eigvals(sys.A)
    lam = lam[np.argsort(-lam.real)]
    return lam[:k] if k is not None else lam

"""The boundary-controlled wave equation of Example ``ex:intro-wave``.

    z_tt = c^2 z_xixi,   z(t,0) = u(t),  z(t,1) = 0,
    y(t) = int_0^1 c_obs(xi) z(t,xi) dxi,

the concrete hyperbolic instance of Pritchard--Salamon Example 4.10. Written as a first-order
system in ``(z, z_t)`` it takes the form ``x' = A x + B u``, ``y = C x`` on

    X = L^2(0,1) x H^{-1}(0,1),      W = H_0^1(0,1) x L^2(0,1),

so that ``W -> X -> V`` is the scale of the introduction and Dirichlet boundary
control is bounded into the coarser space.  The semigroup is a strongly
continuous, nonsmoothing group on ``W`` (unitary in an equivalent
speed-weighted energy norm) and is *not* analytic: this example is precisely
the one the sufficiency theorem for harmonic persistency of excitation
(``thm:analytic-universal-sufficiency``) does not reach, while the
controllability test of ``prop:data-fattorini-hautus`` and the data-driven
regulator of ``thm:lqr-data`` apply unchanged.

``dirichlet``
    control at ``xi = 0`` only.  Every mode has ``b_n = sqrt2 n pi c^2 != 0``,
    so the pair is approximately controllable.

``dirichlet_sym``
    ``z(t,0) = z(t,1) = u(t)``.  A single control cannot reach the modes that
    are antisymmetric about ``xi = 1/2``, so ``b_n = 0`` for every even ``n``:
    the reference *uncontrollable* hyperbolic system, with the obstruction
    ``lambda = pm i n pi c`` available in closed form.

Discretization
--------------
Identical in spirit to :mod:`ddinf.systems.heat`.  Writing ``z_h = Z_I psi_I + u psi_0``
and testing against interior hats gives
``M_II Z_I'' + M_I0 u'' + K_II Z_I + K_I0 u = 0``; row-sum mass lumping makes
``M`` diagonal, so ``M_I0 = 0`` and the ``u''`` term disappears *exactly*.  The
first-order realisation is therefore

    A = [[0, I], [-c^2 M^{-1} K_II, 0]],     B = [[0], [-c^2 M^{-1} K_I0]].

Inner products.  A finite-element function with nodal vector ``a`` has
``||a||_{L^2}^2 = a' M a`` and ``||a||_{H^{-1}}^2 = a' M K^{-1} M a`` (the
discrete Riesz map of the Dirichlet Laplacian), so

    MX = diag(M, M K^{-1} M),        MW = diag(K, M),

which are the discrete inner products of ``X`` and of the energy space ``W``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .fem import Mesh1D, bump, interpolate, mass_matrix, stiffness_matrix
from .base import LinearSystem


def wave_system(kind: str = "dirichlet", *, n_elems: int = 64, speed: float = 1.0,
                obs_center: float = 0.6, obs_width: float = 0.25,
                obs: Callable[[np.ndarray], np.ndarray] | None = None) -> LinearSystem:
    """Semi-discrete wave equation ``z_tt = speed^2 z_xixi`` as a :class:`LinearSystem`.

    The observation is the smooth distributed measurement of the *displacement*,
    ``y(t) = int_0^1 c(xi) z(t,xi) dxi`` with ``c`` the smooth unit-mass bump of
    :func:`ddinf.systems.fem.bump`, as in Pritchard--Salamon Example 4.10.  Its support
    is kept inside ``(0,1)``, so the boundary lift is not observed and the
    output carries no feedthrough term.

    ``speed`` plays the role the diffusivity plays for the heat equation: it
    sets the band ``omega_n = n pi * speed`` the probing input has to cover for
    the record to be informative about the ``n``-th mode.
    """
    if kind not in ("dirichlet", "dirichlet_sym"):
        raise ValueError(f"unknown wave configuration {kind!r}")

    mesh = Mesh1D(n_elems)
    M_full = mass_matrix(mesh, lumped=True)
    K_full = stiffness_matrix(mesh)
    free = np.arange(1, mesh.n_nodes - 1)  # interior nodes
    M = M_full[np.ix_(free, free)]
    K = K_full[np.ix_(free, free)]
    Minv = np.diag(1.0 / np.diag(M))
    c2 = speed ** 2

    if kind == "dirichlet":
        load = K_full[np.ix_(free, [0])]
    else:
        load = K_full[np.ix_(free, [0])] + K_full[np.ix_(free, [mesh.n_nodes - 1])]

    n = free.size
    Z = np.zeros((n, n))
    A = np.block([[Z, np.eye(n)], [-c2 * (Minv @ K), Z]])
    B = np.vstack([np.zeros((n, 1)), -c2 * (Minv @ load)])

    c_fun = obs if obs is not None else bump(obs_center, obs_width)
    c_nodal = interpolate(mesh, c_fun)
    C = np.concatenate([M @ c_nodal[free], np.zeros(n)])[None, :]

    Kinv = np.linalg.inv(K)
    MX = _block_diag(M, M @ Kinv @ M)  # L^2 x H^{-1}
    MW = _block_diag(K, M)  # H_0^1 x L^2 (the energy inner product)

    def expand(X: np.ndarray, u: np.ndarray | float = 0.0) -> np.ndarray:
        """Interior displacement dofs -> full nodal values, with the lift."""
        full = np.zeros((mesh.n_nodes,) + np.shape(X)[1:])
        full[free] = np.asarray(X)[:n]
        full[0] = u
        if kind == "dirichlet_sym":
            full[-1] = u
        return full

    return LinearSystem(
        name=f"wave-{kind}(n={n_elems}, c={speed:g})",
        A=A, B=B, C=C, MX=MX, MW=MW,
        meta={
            "family": "wave",
            "kind": kind,
            "speed": speed,
            "mesh": mesh,
            "free": free,
            "expand": expand,
            "obs_kernel": c_fun,
            "obs_nodal": c_nodal,
            "n_half": n,
        },
    )


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + b.shape[0],) * 2)
    out[: a.shape[0], : a.shape[0]] = a
    out[a.shape[0] :, a.shape[0] :] = b
    return 0.5 * (out + out.T)


def fem_frequencies(sys: LinearSystem, k: int | None = None) -> np.ndarray:
    """Discrete frequencies ``omega`` with ``lambda = pm i omega`` in the spectrum.

    The generator of the semi-discrete wave equation has purely imaginary
    eigenvalues, and these are the exact obstruction candidates of the
    *discretized* system.  With lumped mass they are
    ``omega_{h,n} = (2 c/h) sin(n pi h/2) = n pi c (1 + O(h^2))``.
    """
    omega = np.sort(np.abs(np.linalg.eigvals(sys.A).imag))
    omega = omega[::2]  # eigenvalues come in conjugate pairs
    return omega[:k] if k is not None else omega


def exact_frequencies(k: int, speed: float = 1.0) -> np.ndarray:
    """``omega_n = n pi * speed``, the frequencies of the continuous problem."""
    return np.pi * speed * np.arange(1, k + 1)

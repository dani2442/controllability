"""The retarded equation of Example ``ex:intro-delay``.

    x'(t) = A0 x(t) + A1 x(t-h) + B0 u(t),
    y(t) = e_j' x(t-h),

on the product state space ``X = R^n x L^2(-h,0;R^n)`` carrying the state
``(x(t), x_t)`` with ``x_t(tau) = x(t+tau)``.  This is the ``M = 0`` case of the
neutral family in Pritchard--Salamon Section 4.1 and Example ``ex:intro-delay``;
the delayed output makes ``C`` a
point evaluation of the ``L^2`` component, hence unbounded on ``X`` and bounded
on the finer space ``W``.

Discretisation
--------------
The second component solves the transport equation ``d/dt x_t = d/dtau x_t`` on
``(-h,0)``, whose inflow boundary is ``tau = 0``, where the value is prescribed
by the delay equation itself.  Discretising *that* by P1 finite elements in
``tau`` -- the linear-spline scheme of Banks and Kappel -- is the exact analogue
of the FEM discretisation used for the PDE examples:

    rows 0..N-1 :  M w' = D w          (Galerkin transport, M, D the P1 mass
                                        and derivative matrices on [-h,0])
    row N       :  w_N' = A0 w_N + A1 w_0 + B0 u   (the delay equation)

with nodal unknowns ``w_j(t) ~ x(t + tau_j)``, ``tau_j = -h + j h/N``.  The
inner products are those of ``X`` and ``W``:

    <a,b>_X = a_N . b_N + int_{-h}^0 a b dtau,      <a,b>_W = <a,b>_X + int a' b'.

Closed-form references
----------------------
The characteristic matrix is ``Delta(lambda) = lambda I - A0 - A1 e^{-lambda h}``,
and (Pritchard--Salamon Rmk. 4.1(iv) with ``M=0``) the system is approximately
controllable iff ``rank[Delta(lambda), B0] = n`` for every ``lambda``.  For the
block-triangular example below the obstruction is available in closed form
through the Lambert ``W`` function.
"""

from __future__ import annotations

import numpy as np
from scipy.special import lambertw

from .systems import LinearSystem


def _p1_matrices(n_tau: int, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P1 mass, derivative (``int psi_j psi_k'``) and stiffness on ``[-h, 0]``."""
    d = h / n_tau
    n = n_tau + 1
    M = np.zeros((n, n))
    D = np.zeros((n, n))
    K = np.zeros((n, n))
    m_e = (d / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    d_e = 0.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])  # int psi_j psi_k' on an element
    k_e = (1.0 / d) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(n_tau):
        M[e : e + 2, e : e + 2] += m_e
        D[e : e + 2, e : e + 2] += d_e
        K[e : e + 2, e : e + 2] += k_e
    return M, D, K


def delay_system(A0: np.ndarray, A1: np.ndarray, B0: np.ndarray, *,
                 h: float = 1.0, n_tau: int = 6,
                 obs_component: int = 0) -> LinearSystem:
    """Semi-discrete retarded equation as a :class:`~ddinf.systems.LinearSystem`.

    ``obs_component`` selects which entry of the delayed state ``x(t-h)`` is
    measured; the output operator is a point evaluation in ``tau``.
    """
    A0 = np.atleast_2d(np.asarray(A0, dtype=float))
    A1 = np.atleast_2d(np.asarray(A1, dtype=float))
    B0 = np.atleast_2d(np.asarray(B0, dtype=float))
    n = A0.shape[0]
    N = n_tau
    M, D, K = _p1_matrices(N, h)
    I_n = np.eye(n)

    dim = n * (N + 1)
    E = np.zeros((dim, dim))
    F = np.zeros((dim, dim))
    G = np.zeros((dim, B0.shape[1]))

    # transport rows (all nodes except the inflow node tau = 0)
    E[: n * N] = np.kron(M[:N], I_n)
    F[: n * N] = np.kron(D[:N], I_n)
    # inflow row: the delay equation itself
    E[n * N :, n * N :] = I_n
    F[n * N :, n * N :] = A0
    F[n * N :, :n] = A1
    G[n * N :] = B0

    Einv = np.linalg.inv(E)
    A = Einv @ F
    B = Einv @ G

    C = np.zeros((1, dim))
    C[0, obs_component] = 1.0  # x(t-h), the tau = -h node

    e_last = np.zeros((dim, dim))
    e_last[n * N :, n * N :] = I_n
    MX = np.kron(M, I_n) + e_last  # R^n x L^2(-h,0)
    MW = MX + np.kron(K, I_n)  # H^1 in the delay variable

    return LinearSystem(
        name=f"delay(n={n}, N_tau={N}, h={h:g})",
        A=A, B=B, C=C, MX=MX, MW=MW,
        meta={"family": "delay", "A0": A0, "A1": A1, "B0": B0, "h": h,
              "n_tau": N, "n_states": n, "tau": np.linspace(-h, 0.0, N + 1)},
    )


# ---------------------------------------------------------------- references


def characteristic_matrix(lam: complex, A0: np.ndarray, A1: np.ndarray,
                          h: float) -> np.ndarray:
    """``Delta(lambda) = lambda I - A0 - A1 e^{-lambda h}``."""
    return lam * np.eye(A0.shape[0]) - A0 - A1 * np.exp(-lam * h)


def lambert_roots(d: float, q: float, h: float, *, branches: int = 4) -> np.ndarray:
    """Closed-form roots of the scalar equation ``lambda = d + q e^{-lambda h}``.

    ``lambda = d + W_k(q h e^{-d h}) / h`` over the Lambert branches ``k``.
    This is the exact obstruction of :func:`uncontrollable_pair` -- the
    reference the data-driven test is measured against.
    """
    ks = np.arange(-branches, branches + 1)
    z = lambertw(q * h * np.exp(-d * h), k=ks)
    return np.asarray(d + z / h)


def uncontrollable_pair(h: float = 1.0) -> dict:
    """A block-triangular retarded system whose second mode is unreachable.

    ``B0`` drives only the first coordinate and the second row of ``A0``, ``A1``
    is decoupled from it, so ``eta = (0,1)`` satisfies ``eta' B0 = 0`` and
    ``eta' Delta(lambda) = 0`` at the roots of ``lambda = d + q e^{-lambda h}``.
    """
    A0 = np.array([[-1.0, 0.5], [0.0, -0.5]])
    A1 = np.array([[-0.3, 0.2], [0.0, -0.4]])
    B0 = np.array([[1.0], [0.0]])
    return {"A0": A0, "A1": A1, "B0": B0, "h": h,
            "obstruction": lambert_roots(A0[1, 1], A1[1, 1], h)}


def controllable_pair(h: float = 1.0) -> dict:
    """The same system with the coupling restored, hence approximately controllable."""
    A0 = np.array([[-1.0, 0.5], [0.4, -0.5]])
    A1 = np.array([[-0.3, 0.2], [0.25, -0.4]])
    B0 = np.array([[1.0], [0.0]])
    return {"A0": A0, "A1": A1, "B0": B0, "h": h}


def hautus_delay_defect(lam: complex, A0: np.ndarray, A1: np.ndarray,
                        B0: np.ndarray, h: float) -> float:
    """``sigma_min([Delta(lambda), B0])`` -- zero exactly at an obstruction."""
    blk = np.hstack([characteristic_matrix(lam, A0, A1, h), B0])
    return float(np.linalg.svd(blk, compute_uv=False)[-1])

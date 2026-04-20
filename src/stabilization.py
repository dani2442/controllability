"""Informativity for state-feedback stabilization (Section 6).

Theorem 6.1 (exact input-state data):
    Under im Z_1^{x,u} = R^{n+m}, D_x is informative for stabilization
    by state feedback iff there exist Phi and P with
        P = <x, Phi^T> = P^T > 0,
        (<Dx, Phi^T>)^sym < 0.
    Gain: K = <u, Phi^T> P^{-1}.

Remark 6.2 reduces the problem to the finite LMI
        (A_0 P + B_0 F)^sym < 0,  P > 0,   K = F P^{-1}
on the identified plant.

Theorem 6.3 (noisy input-state data): common quadratic Lyapunov
certificate via the matrix S-lemma. Solves the LMI
        [ beta I_n   Q   L^T  ;
          Q           0    0   ;
          L           0    0  ]
        + alpha * N_x^Phi  <=  0,
    Q > 0,  alpha >= 0,  beta > 0;  K = L Q^{-1}, P = Q^{-1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore

from .identification import identify_state_data
from .lifts import LiftData
from .qmi import InducedQMI, build_induced


@dataclass
class StabilizationResult:
    K: torch.Tensor           # (m, n)
    P: torch.Tensor           # (n, n) common Lyapunov, Q^{-1}
    Q: torch.Tensor           # (n, n) cvxpy variable
    success: bool
    objective: float
    method: str               # "exact" | "noisy"


def stabilize_exact(
    lifts: LiftData,
    *,
    rank_tol: float = 1e-8,
) -> StabilizationResult:
    """Solve the finite LMI in Remark 6.2 from exact state data.

    Uses cvxpy to solve
        min 0  s.t.  P = P^T > 0,  (A0 P + B0 F)^sym < 0,
    where (A0, B0) are the identified matrices.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for LMI-based stabilization.")
    ident = identify_state_data(lifts, rank_tol=rank_tol)
    if not ident.identified:
        raise RuntimeError(
            "Z_1^{x,u} is not surjective (identification failed); cannot "
            "apply the exact-data LMI."
        )
    A0 = ident.A.cpu().numpy().astype(np.float64)
    B0 = ident.B.cpu().numpy().astype(np.float64)
    n = A0.shape[0]
    m = B0.shape[1]

    P = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((m, n))
    eps_P = 1e-6
    eps_Q = 1e-6
    constraints = [
        P - eps_P * np.eye(n) >> 0,
        -(A0 @ P + B0 @ F) - (A0 @ P + B0 @ F).T - eps_Q * np.eye(n) >> 0,
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    success = prob.status in ("optimal", "optimal_inaccurate")
    P_val = torch.tensor(P.value, dtype=lifts.X.dtype) if P.value is not None else torch.eye(n)
    F_val = torch.tensor(F.value, dtype=lifts.X.dtype) if F.value is not None else torch.zeros(m, n)
    K_val = F_val @ torch.linalg.inv(P_val)
    return StabilizationResult(
        K=K_val, P=torch.linalg.inv(P_val), Q=P_val,
        success=success, objective=0.0, method="exact",
    )


def build_Nx_phi(
    *, Pi: torch.Tensor, lifts: LiftData
) -> InducedQMI:
    """Build N_x^Phi from Theorem 6.3 using <Dx, Phi> and Z lifts.

        D_x = <Dx, Phi>, shape (n, N).
        E_x = col(<x, Phi>, <u, Phi>), shape (n + m, N) = Z_1^{x,u} Phi.
    """
    D = lifts.DX
    E = lifts.Z
    return build_induced(Pi, D, E)


def stabilize_noisy(
    *,
    lifts: LiftData,
    Pi: torch.Tensor,
    alpha_lower: float = 0.0,
) -> StabilizationResult:
    """Solve the LMI of Theorem 6.3 for quadratic stabilization.

    Builds N_x^Phi from `Pi` and the state lifts, then solves

        find Q, L, alpha, beta  s.t.  Q > 0, alpha >= `alpha_lower`, beta > 0
            blockdiag(beta I, 0, 0)_offdiag + alpha N_x^Phi <= 0.

    `Pi` must lie in mathbf{Pi}_{n, N} (see `qmi.is_in_Pi_qN`).
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for LMI-based stabilization.")
    n, m = lifts.n, lifts.m
    Ix = build_Nx_phi(Pi=Pi, lifts=lifts).N_mat.cpu().numpy().astype(np.float64)
    # Ix has shape (n + n + m, n + n + m)
    Q = cp.Variable((n, n), symmetric=True)
    L = cp.Variable((m, n))
    alpha = cp.Variable(nonneg=True)
    beta = cp.Variable(pos=True)

    top = cp.bmat([
        [beta * np.eye(n), Q, L.T],
        [Q,                np.zeros((n, n)), np.zeros((n, m))],
        [L,                np.zeros((m, n)), np.zeros((m, m))],
    ])
    eps_Q = 1e-6
    constraints = [
        Q - eps_Q * np.eye(n) >> 0,
        alpha >= alpha_lower,
        -top - alpha * Ix >> 0,
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    success = prob.status in ("optimal", "optimal_inaccurate")
    Q_val = torch.tensor(Q.value, dtype=lifts.X.dtype) if Q.value is not None else torch.eye(n)
    L_val = torch.tensor(L.value, dtype=lifts.X.dtype) if L.value is not None else torch.zeros(m, n)
    K = L_val @ torch.linalg.inv(Q_val)
    return StabilizationResult(
        K=K, P=torch.linalg.inv(Q_val), Q=Q_val,
        success=success, objective=float(prob.value or 0.0), method="noisy",
    )


__all__ = [
    "StabilizationResult",
    "stabilize_exact",
    "stabilize_noisy",
    "build_Nx_phi",
]

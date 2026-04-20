"""Informativity for dissipativity (Section 8).

Theorem 8.1 (exact data): Under inertia(S) = (p, 0, m) the data is
informative for dissipativity iff
    (i)  im Z_1^{x,u} = R^{n+m}, and
    (ii) there exists P >= 0 such that (8.4) holds on Phi-pairings.

Theorem 8.3 (noisy data): LMI (8.8) with the induced QMI N_{xy}^Phi.

Both theorems reduce to LMIs on the identified / consistent plants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore

from .identification import identify_state_output_data
from .lifts import LiftData
from .qmi import build_induced


@dataclass
class DissipativityResult:
    P: torch.Tensor
    success: bool
    method: str  # "exact" | "noisy"
    objective: float = 0.0


def _supply_blocks(S: torch.Tensor, m: int, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S_np = S.cpu().numpy().astype(np.float64)
    if S_np.shape != (m + p, m + p):
        raise ValueError(f"S must be (m+p, m+p)=({m + p}, {m + p}); got {S_np.shape}.")
    # Partition S into blocks along u / y, matching the paper's convention.
    return S_np, S_np[:m, :m], S_np[:m, m:], S_np[m:, m:], S_np[m:, :m]


def dissipativity_exact(
    lifts: LiftData,
    y_lift_0: torch.Tensor,
    S: torch.Tensor,
    *,
    rank_tol: float = 1e-8,
) -> DissipativityResult:
    """Check exact-data informativity for dissipativity wrt supply rate S.

    Identifies the unique plant (A0, B0, C0, D0) and solves the LMI
        [I 0; A B]^T [0 P; P 0] [I 0; A B]
        - [0 I; C D]^T S [0 I; C D]  <=  0,   P >= 0.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for LMI-based dissipativity.")
    ident = identify_state_output_data(lifts, y_lift_0, rank_tol=rank_tol)
    if not ident.identified:
        raise RuntimeError("Data is not informative for identification.")
    A = ident.A.cpu().numpy().astype(np.float64)
    B = ident.B.cpu().numpy().astype(np.float64)
    C = ident.C.cpu().numpy().astype(np.float64)
    D = ident.D.cpu().numpy().astype(np.float64)
    S_np = S.cpu().numpy().astype(np.float64)
    n, m, p = lifts.n, lifts.m, lifts.p

    P = cp.Variable((n, n), symmetric=True)

    # M1 = [I 0; A B]^T [0 P; P 0] [I 0; A B] = [A^T P + P A,  P B; B^T P, 0]
    M1 = cp.bmat([
        [A.T @ P + P @ A, P @ B],
        [B.T @ P,         np.zeros((m, m))],
    ])
    # block = [0_{m x n}  I_m; C  D]  of shape (m + p, n + m)
    block_top = np.concatenate([np.zeros((m, n)), np.eye(m)], axis=1)
    block_bot = np.concatenate([C, D], axis=1)
    block = np.concatenate([block_top, block_bot], axis=0)
    M2 = block.T @ S_np @ block

    constraints = [P >> 0, -(M1 - M2) >> 0]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    success = prob.status in ("optimal", "optimal_inaccurate")
    dtype = ident.A.dtype
    device = ident.A.device
    P_val = torch.tensor(P.value, dtype=dtype, device=device) if P.value is not None else torch.eye(n, dtype=dtype, device=device)
    return DissipativityResult(P=P_val, success=success, method="exact")


def dissipativity_noisy(
    *,
    lifts: LiftData,
    y_lift_0: torch.Tensor,
    S: torch.Tensor,
    Pi: torch.Tensor,
    alpha_lower: float = 0.0,
) -> DissipativityResult:
    """Solve Theorem 8.3's LMI.

    The induced matrix is N_{xy}^Phi = N_Pi(D_xy, E_xy) where
        D_xy = col(<Dx, Phi>, <y, Phi>),      shape (n + p, N)
        E_xy = col(<x, Phi>, <u, Phi>),       shape (n + m, N)
    and `Pi` belongs to mathbf{Pi}_{n+p, n+m}.

    The LMI to solve (paper eq. 8.8) is built in its 4x4 block form with
    inverse supply `-S^{-1} = [[hatF, hatG]; [hatG^T, hatH]]`.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for LMI-based dissipativity.")
    n, m, p = lifts.n, lifts.m, lifts.p
    D_mat = torch.cat([lifts.DX, y_lift_0], dim=0)  # (n + p, N)
    E_mat = torch.cat([lifts.X, lifts.U[:m]], dim=0)  # (n + m, N)  (use <u, Phi>)
    Nxy = build_induced(Pi, D_mat, E_mat).N_mat.cpu().numpy().astype(np.float64)

    S_np = S.cpu().numpy().astype(np.float64)
    S_inv = np.linalg.inv(S_np)
    hatF = -S_inv[:m, :m]
    hatG = -S_inv[:m, m:]
    hatH = -S_inv[m:, m:]

    Q = cp.Variable((n, n), symmetric=True)
    alpha = cp.Variable(nonneg=True)

    block = cp.bmat([
        [Q,                np.zeros((n, p)), np.zeros((n, n)), np.zeros((n, m))],
        [np.zeros((p, n)), hatH,              np.zeros((p, n)), -hatG.T],
        [np.zeros((n, n)), np.zeros((n, p)), -Q,                np.zeros((n, m))],
        [np.zeros((m, n)), -hatG,             np.zeros((m, n)), hatF],
    ])
    constraints = [
        Q >> 1e-6 * np.eye(n),
        alpha >= alpha_lower,
        block - alpha * Nxy >> 0,
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    success = prob.status in ("optimal", "optimal_inaccurate")
    dtype = lifts.X.dtype
    device = lifts.X.device
    Q_val = torch.tensor(Q.value, dtype=dtype, device=device) if Q.value is not None else torch.eye(n, dtype=dtype, device=device)
    return DissipativityResult(
        P=torch.linalg.inv(Q_val), success=success, method="noisy",
        objective=float(prob.value or 0.0),
    )


__all__ = [
    "DissipativityResult",
    "dissipativity_exact",
    "dissipativity_noisy",
]

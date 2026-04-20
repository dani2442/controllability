"""Informativity for continuous-time LQR (Section 7).

Theorem 7.1: under im Z_1^{x,u} = R^{n+m}, the data D_x is informative
for optimal LQR iff (A_0, B_0) is stabilizable and (A_0, Q^{1/2}) is
detectable, where (A_0, B_0) is the unique identified plant. The common
optimal gain is
        K_* = -R^{-1} B_0^T P_*,
with P_* the unique stabilizing solution of the CARE
        A_0^T P + P A_0 - P B_0 R^{-1} B_0^T P + Q = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from .identification import identify_state_data
from .lifts import LiftData


@dataclass
class LQRResult:
    A0: torch.Tensor
    B0: torch.Tensor
    P: torch.Tensor
    K: torch.Tensor
    informative: bool


def is_stabilizable(A: np.ndarray, B: np.ndarray, *, tol: float = 1e-8) -> bool:
    """PBH test: [A - lam I, B] has rank n for every lam with Re(lam) >= 0."""
    n = A.shape[0]
    eigs = np.linalg.eigvals(A)
    for lam in eigs:
        if lam.real >= -tol:
            M = np.concatenate([A - lam * np.eye(n), B], axis=1)
            if np.linalg.matrix_rank(M, tol=tol) < n:
                return False
    return True


def is_detectable(A: np.ndarray, C: np.ndarray, *, tol: float = 1e-8) -> bool:
    return is_stabilizable(A.T, C.T, tol=tol)


def solve_lqr_exact(
    lifts: LiftData,
    Q: torch.Tensor,
    R: torch.Tensor,
    *,
    rank_tol: float = 1e-8,
) -> LQRResult:
    """Data-driven CT-LQR on identified plant.

    Args:
        lifts: state-data LiftData. Z must be surjective.
        Q: (n, n) state weighting, PSD.
        R: (m, m) input weighting, PD.

    Returns:
        LQRResult with optimal gain K = -R^{-1} B_0^T P_*.
    """
    ident = identify_state_data(lifts, rank_tol=rank_tol)
    if not ident.identified:
        raise RuntimeError("Data is not informative for identification.")
    A_np = ident.A.cpu().numpy().astype(np.float64)
    B_np = ident.B.cpu().numpy().astype(np.float64)
    Q_np = Q.cpu().numpy().astype(np.float64)
    R_np = R.cpu().numpy().astype(np.float64)

    if Q_np.shape[0] == 0:
        raise ValueError("Q must be n x n with n > 0.")
    # Detectability of (A, Q^{1/2})
    Qh = _principal_sqrtm(Q_np)
    inform = is_stabilizable(A_np, B_np) and is_detectable(A_np, Qh)

    P_np = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K_np = -np.linalg.solve(R_np, B_np.T @ P_np)

    dtype = ident.A.dtype
    device = ident.A.device
    return LQRResult(
        A0=ident.A, B0=ident.B,
        P=torch.tensor(P_np, dtype=dtype, device=device),
        K=torch.tensor(K_np, dtype=dtype, device=device),
        informative=inform,
    )


def _principal_sqrtm(M: np.ndarray) -> np.ndarray:
    """Square root of a symmetric PSD matrix via spectral decomposition."""
    M_sym = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M_sym)
    w_clip = np.clip(w, a_min=0.0, a_max=None)
    return (V * np.sqrt(w_clip)) @ V.T


__all__ = ["LQRResult", "solve_lqr_exact", "is_stabilizable", "is_detectable"]

"""QMI (quadratic matrix inequality) helpers from Section 2.

Implements the notation of Van Waarde et al. (2023) used throughout the
noisy-data theorems in Sections 5 and 7 of the paper:

    Z_N(Pi)    = { V in R^{q x N} :
                     [ I_q ;  V^T ]^T Pi [ I_q ;  V^T ]  >=  0 }
    Pi_{q,N}   = { Pi in S^{q+N} : Pi_22 <= 0,
                                   Pi | Pi_22 >= 0,
                                   ker Pi_22 ⊆ ker Pi_12 }
    N_Pi(D, E) = [ I  D ;  0  -E ]  Pi  [ I  D ;  0  -E ]^T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def schur_complement_22(Pi: torch.Tensor, q: int) -> torch.Tensor:
    """Generalized Schur complement Pi | Pi_22 = Pi_11 - Pi_12 Pi_22^dagger Pi_12^T."""
    if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
        raise ValueError("Pi must be a square matrix.")
    Pi11 = Pi[:q, :q]
    Pi12 = Pi[:q, q:]
    Pi22 = Pi[q:, q:]
    return Pi11 - Pi12 @ torch.linalg.pinv(Pi22) @ Pi12.T


def is_in_Pi_qN(Pi: torch.Tensor, q: int, *, atol: float = 1e-10) -> bool:
    """Check Pi in mathbf{Pi}_{q,N}. Mostly used to validate inputs."""
    N = Pi.shape[0] - q
    Pi22 = Pi[q:, q:]
    Pi12 = Pi[:q, q:]
    # Pi_22 <= 0
    eig22 = torch.linalg.eigvalsh(0.5 * (Pi22 + Pi22.T))
    if eig22.max().item() > atol:
        return False
    # Schur >= 0
    Sc = schur_complement_22(Pi, q)
    eig_sc = torch.linalg.eigvalsh(0.5 * (Sc + Sc.T))
    if eig_sc.min().item() < -atol:
        return False
    # ker Pi_22 subset ker Pi_12: check (I - Pi22 Pi22^dagger) * Pi_12^T == 0
    pinv22 = torch.linalg.pinv(Pi22)
    kerproj = torch.eye(N, dtype=Pi.dtype, device=Pi.device) - Pi22 @ pinv22
    residual = Pi12 @ kerproj
    return residual.abs().max().item() <= atol * max(1.0, Pi12.abs().max().item())


def induced_qmi_matrix(
    Pi: torch.Tensor, D: torch.Tensor, E: torch.Tensor
) -> torch.Tensor:
    r"""Compute N_Pi(D, E) (equation (2.6) of the preliminaries).

    N_Pi(D, E) = [ I  D ;  0  -E ] Pi [ I  D ;  0  -E ]^T
    with I an identity block of size equal to the row count of D.
    """
    if D.ndim != 2 or E.ndim != 2:
        raise ValueError("D and E must be matrices.")
    q = D.shape[0]
    N = D.shape[1]
    if E.shape[1] != N:
        raise ValueError("D and E must share N columns.")
    r = E.shape[0]
    I_q = torch.eye(q, dtype=Pi.dtype, device=Pi.device)
    M = torch.zeros(q + r, q + N, dtype=Pi.dtype, device=Pi.device)
    M[:q, :q] = I_q
    M[:q, q:] = D
    M[q:, q:] = -E
    return M @ Pi @ M.T


def make_exact_Pi(q: int, N: int, *, dtype: torch.dtype = torch.float64,
                  device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return Pi = diag(0_q, I_N) flipped to the `zero-noise' convention.

    With Pi_11 = 0 and Pi_22 = 0 the QMI becomes the exact equation
    D = Theta E; this is what the paper calls the "noise vanishes" case.
    However, to stay inside mathbf{Pi}_{q,N} (which requires Pi_22 <= 0),
    we use
        Pi = [[0, 0], [0, -I_N]]
    which gives Z_N(Pi) = {V : V V^T <= 0} = {0}, so the induced
    constraint D - Theta E = 0 is imposed. This is the explicit form used
    in Remark 5.6 of the stabilization section.
    """
    Pi = torch.zeros(q + N, q + N, dtype=dtype, device=device)
    Pi[q:, q:] = -torch.eye(N, dtype=dtype, device=device)
    return Pi


def ball_Pi(q: int, N: int, noise_bound: float,
            *, dtype: torch.dtype = torch.float64,
            device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Spherical-noise bound: Pi = diag(bound^2 * I_q, -I_N).

    Corresponds to the assumption  W W^T <= bound^2 * I_q  on the noise
    matrix. This Pi lies in mathbf{Pi}_{q,N} whenever bound > 0.
    """
    Pi = torch.zeros(q + N, q + N, dtype=dtype, device=device)
    Pi[:q, :q] = (noise_bound ** 2) * torch.eye(q, dtype=dtype, device=device)
    Pi[q:, q:] = -torch.eye(N, dtype=dtype, device=device)
    return Pi


@dataclass
class InducedQMI:
    """Holds the induced QMI data matrix alongside block slicing.

    For convenience the caller can unpack
        N11, N12, N22, q, r
    so that the Schur complement and the LMI in Theorem 5.3 can be built
    without recomputing dimensions.
    """

    N_mat: torch.Tensor
    q: int
    r: int

    @property
    def N11(self) -> torch.Tensor:
        return self.N_mat[: self.q, : self.q]

    @property
    def N12(self) -> torch.Tensor:
        return self.N_mat[: self.q, self.q:]

    @property
    def N22(self) -> torch.Tensor:
        return self.N_mat[self.q:, self.q:]

    def schur22(self) -> torch.Tensor:
        return schur_complement_22(self.N_mat, self.q)


def build_induced(
    Pi: torch.Tensor, D: torch.Tensor, E: torch.Tensor
) -> InducedQMI:
    return InducedQMI(
        N_mat=induced_qmi_matrix(Pi, D, E),
        q=D.shape[0],
        r=E.shape[0],
    )


__all__ = [
    "schur_complement_22",
    "is_in_Pi_qN",
    "induced_qmi_matrix",
    "make_exact_Pi",
    "ball_Pi",
    "InducedQMI",
    "build_induced",
]

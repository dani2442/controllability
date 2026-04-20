"""Informativity for identification (Section 4 of the paper).

Theorem 4.1 (state data):
    Sigma_{D_x} = {(A*, B*)}  iff  im Z_1^{x,u} = R^{n+m}.
    The unique consistent pair is [A* B*] = <Dx, Phi> R_x where
    R_x : R^{n+m} -> D(I) is any linear right inverse of Z_1^{x,u}.

Theorem 4.2 (I/O data, up to similarity):
    With (A*, B*, C*, D*) minimal and the hypotheses of Theorem 2.4,
    every minimal system in Sigma_{D_y} is similar to (A*, B*, C*, D*).

Numerically we realize R_x as a pseudoinverse: with lifts evaluated on a
finite basis Phi, Z_1^{x,u} collapses to a matrix `Z in R^{(n+m) x N}`,
and we compute `[A B] = <Dx, Phi> Z^+`. Proposition 2.2 shows the result
does not depend on the choice of right inverse whenever Z is surjective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .lifts import LiftData


@dataclass
class IdentificationResult:
    A: torch.Tensor         # (n, n)
    B: torch.Tensor         # (n, m)
    C: torch.Tensor | None  # (p, n) when y is available
    D: torch.Tensor | None  # (p, m) when y is available
    identified: bool        # im Z reached full rank
    rank_Z: int
    singular_values_Z: torch.Tensor
    residual_state: float   # || <Dx, Phi> - [A B] Z ||_F
    residual_output: float  # analogous for <y, Phi> - [C D] Z


def _pseudoinverse(mat: torch.Tensor, rcond: float) -> torch.Tensor:
    return torch.linalg.pinv(mat, rcond=rcond)


def check_identifiability(
    lifts: LiftData, *, rank_tol: float = 1e-8
) -> Tuple[bool, int, torch.Tensor]:
    """Test whether Z_1^{x,u} Phi has full row rank n+m (Theorem 4.1 (ii))."""
    Z = lifts.Z  # (n+m, N)
    svals = torch.linalg.svdvals(Z)
    if svals.numel() == 0:
        return False, 0, svals
    tol = rank_tol * svals.max()
    rank = int((svals > tol).sum().item())
    return rank == Z.shape[0], rank, svals


def identify_state_data(
    lifts: LiftData, *, rank_tol: float = 1e-8
) -> IdentificationResult:
    """Recover (A, B) from state data.

    Under `im Z = R^{n+m}`, Proposition 2.2 yields
        [A  B] = <Dx, Phi> Z^+,
    independent of the particular right inverse.
    """
    Z = lifts.Z  # (n+m, N)
    DX = lifts.DX  # (n, N)

    identified, rank, svals = check_identifiability(lifts, rank_tol=rank_tol)
    rcond = rank_tol
    AB = DX @ _pseudoinverse(Z, rcond=rcond)  # (n, n+m)
    A = AB[:, : lifts.n]
    B = AB[:, lifts.n:]
    residual = torch.linalg.matrix_norm(DX - AB @ Z, ord="fro").item()

    return IdentificationResult(
        A=A, B=B, C=None, D=None,
        identified=identified,
        rank_Z=rank,
        singular_values_Z=svals,
        residual_state=residual,
        residual_output=0.0,
    )


def identify_state_output_data(
    lifts: LiftData,
    y_lift_0: torch.Tensor,
    *,
    rank_tol: float = 1e-8,
) -> IdentificationResult:
    """Recover (A, B, C, D) from input-state-output data.

    `y_lift_0` is <y, Phi> (i.e. Y_1^y Phi) of shape (p, N). Same right
    inverse of Z gives
        [C  D] = <y, Phi> Z^+.
    """
    Z = lifts.Z
    identified, rank, svals = check_identifiability(lifts, rank_tol=rank_tol)
    rcond = rank_tol
    Zinv = _pseudoinverse(Z, rcond=rcond)
    AB = lifts.DX @ Zinv
    CD = y_lift_0 @ Zinv
    n, m = lifts.n, lifts.m
    A, B = AB[:, :n], AB[:, n:]
    C, D = CD[:, :n], CD[:, n:]
    resid_state = torch.linalg.matrix_norm(lifts.DX - AB @ Z, ord="fro").item()
    resid_output = torch.linalg.matrix_norm(y_lift_0 - CD @ Z, ord="fro").item()
    return IdentificationResult(
        A=A, B=B, C=C, D=D,
        identified=identified,
        rank_Z=rank,
        singular_values_Z=svals,
        residual_state=resid_state,
        residual_output=resid_output,
    )


__all__ = [
    "IdentificationResult",
    "check_identifiability",
    "identify_state_data",
    "identify_state_output_data",
]

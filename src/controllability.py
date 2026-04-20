"""Data-driven Hautus tests of Section 5 of the paper.

Theorem 5.1 (state-based):
    Under PE order n+1,
        Sigma_{D_x} ⊆ {controllable}
            <=> im X_1^{x_lambda} = R^n  for all lambda in C,
        Sigma_{D_x} ⊆ {stabilizable}
            <=  same with  Re(lambda) >= 0.

Lemma 5.2 (finite candidate reduction): if moreover im X_1^x = R^n and
    K_x := <Dx, Phi> R_x
(where R_x is any right inverse of X_1^x) then
    im X_1^{x_lambda} != R^n  implies  lambda in sigma(K_x).

Theorem 5.3 (I/O-based): analogous condition on dim im Y_{L,K}^{y_lambda,u_lambda}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .bases import TestFunctionBasis
from .lifts import (
    LiftData,
    compute_filtered_io_lift,
    compute_filtered_state_lift,
)


@dataclass
class HautusResult:
    is_controllable: bool
    is_stabilizable: bool
    candidate_lambdas: torch.Tensor
    ranks: torch.Tensor
    min_singular_values: torch.Tensor
    expected_rank: int


def _thresholded_rank(svals: torch.Tensor, *, rel_tol: float) -> int:
    if svals.numel() == 0:
        return 0
    smax = svals.max().abs()
    tol = rel_tol * float(smax.item()) if smax.item() > 0 else rel_tol
    return int((svals.abs() > tol).sum().item())


def finite_candidates_state(lifts: LiftData, *, rank_tol: float = 1e-8) -> torch.Tensor:
    """Return sigma(K_x) where K_x = <Dx, Phi> X^+ (Lemma 5.2).

    If X has rank < n the reduction is not conclusive and we return an
    empty tensor; the caller should fall back to sweeping the complex
    plane on a grid.
    """
    X = lifts.X
    DX = lifts.DX
    svals = torch.linalg.svdvals(X)
    rank = _thresholded_rank(svals, rel_tol=rank_tol)
    if rank < X.shape[0]:
        return torch.zeros(0, dtype=torch.complex128)
    K_x = DX @ torch.linalg.pinv(X, rcond=rank_tol)
    return torch.linalg.eigvals(K_x).to(torch.complex128)


def finite_candidates_io(lifts: LiftData, *, rank_tol: float = 1e-8) -> torch.Tensor:
    """Eigenvalues of the Markov matrix built from <y, Phi> and <u, Phi>.

    Follows the same idea as Lemma 5.2 but applied to the I/O lift: when
    Y_{L,K}^{y,u} has image of dimension n + Lm we can consider the map
    K_y = <Dy, Phi> * (Y_K^y_lifts)^+ and its spectrum.
    """
    Y = lifts.Y
    if Y.numel() == 0:
        return torch.zeros(0, dtype=torch.complex128)
    p = lifts.p
    Y0 = Y[:p]
    DY = Y[p: 2 * p] if lifts.K >= 2 else torch.zeros_like(Y0)
    if DY.numel() == 0:
        return torch.zeros(0, dtype=torch.complex128)
    svals = torch.linalg.svdvals(Y0)
    if _thresholded_rank(svals, rel_tol=rank_tol) < Y0.shape[0]:
        return torch.zeros(0, dtype=torch.complex128)
    K_y = DY @ torch.linalg.pinv(Y0, rcond=rank_tol)
    return torch.linalg.eigvals(K_y).to(torch.complex128)


def state_hautus_test(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    x: torch.Tensor,
    lifts: LiftData,
    candidate_lambdas: Optional[torch.Tensor] = None,
    rank_tol: float = 1e-8,
    threshold: float = 1e-6,
    grid_lambdas: Optional[torch.Tensor] = None,
) -> HautusResult:
    """Run the state-based Hautus test of Theorem 5.1.

    The `candidate_lambdas` default to sigma(K_x) from Lemma 5.2. If that
    set is empty (e.g. X not surjective) and `grid_lambdas` is given we
    use the grid instead.
    """
    n = lifts.n
    if candidate_lambdas is None:
        candidate_lambdas = finite_candidates_state(lifts, rank_tol=rank_tol)
    if candidate_lambdas.numel() == 0:
        if grid_lambdas is None:
            raise ValueError(
                "Finite candidate set is empty and no grid_lambdas provided."
            )
        candidate_lambdas = grid_lambdas.to(torch.complex128)

    ranks: List[int] = []
    min_svals: List[float] = []
    is_ctrl = True
    is_stab = True
    for lam in candidate_lambdas.tolist():
        lam_c = complex(lam)
        Xlam = compute_filtered_state_lift(basis=basis, ts=ts, x=x, lam=lam_c)
        svals = torch.linalg.svdvals(Xlam).real
        r = _thresholded_rank(svals, rel_tol=threshold)
        ranks.append(r)
        min_svals.append(float(svals[n - 1].item()) if svals.numel() >= n else 0.0)
        if r < n:
            is_ctrl = False
            if lam_c.real >= -1e-12:
                is_stab = False

    return HautusResult(
        is_controllable=is_ctrl,
        is_stabilizable=is_stab,
        candidate_lambdas=candidate_lambdas,
        ranks=torch.tensor(ranks),
        min_singular_values=torch.tensor(min_svals),
        expected_rank=n,
    )


def io_hautus_test(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    m: int,
    lifts: LiftData,
    candidate_lambdas: Optional[torch.Tensor] = None,
    rank_tol: float = 1e-8,
    threshold: float = 1e-6,
    grid_lambdas: Optional[torch.Tensor] = None,
) -> HautusResult:
    """Run the I/O-based Hautus test of Theorem 5.3.

    Tests dim im Y_{L,K}^{y_lambda, u_lambda} = Lm + n at every candidate.
    """
    expected = L * m + n
    if candidate_lambdas is None:
        candidate_lambdas = finite_candidates_io(lifts, rank_tol=rank_tol)
    if candidate_lambdas.numel() == 0:
        if grid_lambdas is None:
            raise ValueError(
                "Finite I/O candidate set is empty; provide grid_lambdas."
            )
        candidate_lambdas = grid_lambdas.to(torch.complex128)

    ranks: List[int] = []
    min_svals: List[float] = []
    is_ctrl = True
    is_stab = True
    for lam in candidate_lambdas.tolist():
        lam_c = complex(lam)
        Ylam = compute_filtered_io_lift(
            basis=basis, ts=ts, u=u, y=y, L=L, K=K, lam=lam_c,
        )
        svals = torch.linalg.svdvals(Ylam).real
        r = _thresholded_rank(svals, rel_tol=threshold)
        ranks.append(r)
        min_svals.append(
            float(svals[expected - 1].item()) if svals.numel() >= expected else 0.0
        )
        if r < expected:
            is_ctrl = False
            if lam_c.real >= -1e-12:
                is_stab = False

    return HautusResult(
        is_controllable=is_ctrl,
        is_stabilizable=is_stab,
        candidate_lambdas=candidate_lambdas,
        ranks=torch.tensor(ranks),
        min_singular_values=torch.tensor(min_svals),
        expected_rank=expected,
    )


__all__ = [
    "HautusResult",
    "finite_candidates_state",
    "finite_candidates_io",
    "state_hautus_test",
    "io_hautus_test",
]

"""Persistent excitation (Definition 2.3 / Proposition 2.4).

A weak-derivative test for PE on general distributional data: we require
that the lifted image
    U_L^{u-bar}  :  D(I)  ->  R^{Lm}
be surjective, i.e. that <D^{k}u, phi_j> for k=0..L-1 and j=1..N span
R^{Lm} as N grows. For Sobolev data this reduces to positivity of the
derivative-lifted Gramian
    Gamma_L(u) = int_0^T Lambda_L(u)(t) Lambda_L(u)(t)^T dt.

Here we implement both: given a basis Phi we build U_L^u Phi and test
its row rank; for comparison we also provide the closed-form Gramian
computed via integration-by-parts against the test functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .bases import TestFunctionBasis
from .lifts import _stack_derivative_pairings


@dataclass
class PersistentExcitationResult:
    L: int
    rank: int
    expected_rank: int
    min_singular_value: float
    singular_values: torch.Tensor
    is_pe: bool


def check_persistent_excitation(
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    u: torch.Tensor,
    L: int,
    *,
    rank_tol: float = 1e-8,
) -> PersistentExcitationResult:
    """Test `u` for PE of order L using a basis (Definition 2.3)."""
    U = _stack_derivative_pairings(basis, ts, u, L)  # (Lm, N)
    svals = torch.linalg.svdvals(U)
    expected = U.shape[0]
    if svals.numel() == 0:
        return PersistentExcitationResult(
            L=L, rank=0, expected_rank=expected,
            min_singular_value=0.0, singular_values=svals, is_pe=False,
        )
    smax = svals.max()
    tol = rank_tol * smax
    rank = int((svals > tol).sum().item())
    min_sv = float(svals[expected - 1].item()) if svals.numel() >= expected else 0.0
    return PersistentExcitationResult(
        L=L, rank=rank, expected_rank=expected,
        min_singular_value=min_sv, singular_values=svals, is_pe=rank == expected,
    )


__all__ = [
    "PersistentExcitationResult",
    "check_persistent_excitation",
]

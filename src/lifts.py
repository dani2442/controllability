"""Weak lift operators from `sections/preliminaries.tex`.

Given a test-function basis Phi = (phi_1, ..., phi_N), the paper's weak
lifts are linear maps with values in finite-dimensional real spaces:

    U_L^u Phi      = col( <u, Phi>, <Du, Phi>, ..., <D^{L-1}u, Phi> )
                   in R^{Lm x N}
    X_K^x Phi      = col( <x, Phi>, <Dx, Phi>, ..., <D^{K-1}x, Phi> )
                   in R^{Kn x N}
    Y_K^y Phi      = col( <y, Phi>, <Dy, Phi>, ..., <D^{K-1}y, Phi> )
                   in R^{Kp x N}
    Z_L^{x,u} Phi  = col( X_1^x Phi, U_L^u Phi )            in R^{(n+Lm) x N}
    Y_{L,K}^{y,u}  = col( Y_K^y Phi, U_L^u Phi )            in R^{(Kp+Lm) x N}

For data living in H^{L-1}/H^{K-1}, pairings involving weak derivatives
are computed via integration by parts against the smooth test functions.
See `bases.pair`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bases import TestFunctionBasis, pair


@dataclass
class LiftData:
    """Stacked pairings computed from a trajectory and a test-function basis."""

    U: torch.Tensor  # (Lm, N)
    X: torch.Tensor  # (n,  N)   always X_1
    DX: torch.Tensor  # (n,  N)   <Dx, Phi>
    Y: torch.Tensor  # (Kp, N)   full Y_K (only present for IO data)
    L: int
    K: int
    n: int
    m: int
    p: int

    @property
    def Z(self) -> torch.Tensor:
        """Z_L^{x,u} Phi = [X_1^x Phi ; U_L^u Phi] of shape (n + Lm, N)."""
        return torch.cat([self.X, self.U], dim=0)

    @property
    def Y_LK(self) -> torch.Tensor:
        """Y_{L,K}^{y,u} Phi = [Y_K^y Phi ; U_L^u Phi] of shape (Kp + Lm, N)."""
        return torch.cat([self.Y, self.U], dim=0)


def _stack_derivative_pairings(
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    signal: torch.Tensor,
    orders: int,
) -> torch.Tensor:
    """Return the row-block [<D^k signal, Phi> for k in 0..orders-1]."""
    parts = []
    for k in range(orders):
        parts.append(pair(basis, ts, signal, order=k))  # (d, N)
    return torch.cat(parts, dim=0)  # (orders*d, N)


def compute_state_lifts(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    L: int,
    K: int = 1,
) -> LiftData:
    """Build all pairings required for the state-data theorems.

    Returns X_1^x Phi, <Dx, Phi>, U_L^u Phi (and allocates an empty Y).
    """
    if L < 1 or K < 1:
        raise ValueError("L, K must be >= 1.")
    if u.ndim != 2 or x.ndim != 2:
        raise ValueError("u and x must be 2-D (N_time, d).")
    if u.shape[0] != x.shape[0] or u.shape[0] != ts.shape[0]:
        raise ValueError("u, x, ts must share N_time.")

    n = x.shape[1]
    m = u.shape[1]
    X = pair(basis, ts, x, order=0)  # (n, N)
    DX = pair(basis, ts, x, order=1)  # (n, N)  integration by parts
    U = _stack_derivative_pairings(basis, ts, u, L)  # (Lm, N)
    Y = torch.zeros(0, U.shape[1], dtype=U.dtype, device=U.device)
    return LiftData(U=U, X=X, DX=DX, Y=Y, L=L, K=K, n=n, m=m, p=0)


def compute_io_lifts(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor | None = None,
    n: int,
    L: int,
    K: int,
) -> LiftData:
    """Build pairings for the I/O theorems.

    If `x` is provided (input-state-output case) the lift also carries X
    and DX; otherwise these are empty.
    """
    if L < K or K < 1:
        raise ValueError("Need L >= K >= 1.")
    m = u.shape[1]
    p = y.shape[1]
    U = _stack_derivative_pairings(basis, ts, u, L)  # (Lm, N)
    Y = _stack_derivative_pairings(basis, ts, y, K)  # (Kp, N)

    if x is not None:
        X = pair(basis, ts, x, order=0)
        DX = pair(basis, ts, x, order=1)
    else:
        X = torch.zeros(n, U.shape[1], dtype=U.dtype, device=U.device)
        DX = torch.zeros_like(X)

    return LiftData(U=U, X=X, DX=DX, Y=Y, L=L, K=K, n=n, m=m, p=p)


def compute_filtered_state_lift(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    x: torch.Tensor,
    lam: complex,
) -> torch.Tensor:
    """Compute X_1^{x_lambda} Phi where x_lambda = Dx - lambda x.

    Uses integration by parts for the Dx piece. Returns a complex matrix
    of shape (n, N). The result is the weak lift used by Theorem 4.1
    (state-based data-driven Hautus test).
    """
    DX = pair(basis, ts, x, order=1)
    X = pair(basis, ts, x, order=0)
    lam_t = torch.tensor(lam, dtype=torch.complex128)
    DX_c = DX.to(torch.complex128)
    X_c = X.to(torch.complex128)
    return DX_c - lam_t * X_c


def compute_filtered_io_lift(
    *,
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
) -> torch.Tensor:
    """Compute Y_{L,K}^{y_lambda, u_lambda} Phi used by Theorem 4.3.

    Because the basis is smooth, we form the weak lambda-filter by moving
    derivatives onto the test functions and rescaling:
        <D(f) - lambda f, phi> = -<f, D phi + lambda phi>.
    """
    device = u.device
    dtype = u.dtype
    N_time = ts.shape[0]
    phi0 = basis.evaluate(ts, order=0)  # (N_time, N)
    phi1 = basis.evaluate(ts, order=1)  # (N_time, N)

    lam_c = complex(lam)
    # Build "filtered" test functions in the distributional sense:
    #   <Df - lam f, phi> = -<f, phi' + lam phi>  (IBP, boundary terms zero).
    # We keep pairings real whenever lam is real to avoid complex promotion.
    if abs(lam_c.imag) < 1e-15:
        lam_real = float(lam_c.real)
        filt = -(phi1 + lam_real * phi0)  # (N_time, N)
        u_lam_pairs_0 = u.T @ (filt * _trap_weights(ts).unsqueeze(1))
        y_lam_pairs_0 = y.T @ (filt * _trap_weights(ts).unsqueeze(1))
    else:
        phi0c = phi0.to(torch.complex128)
        phi1c = phi1.to(torch.complex128)
        filt = -(phi1c + lam_c * phi0c)
        w = _trap_weights(ts).to(torch.complex128).unsqueeze(1)
        u_lam_pairs_0 = u.to(torch.complex128).T @ (filt * w)
        y_lam_pairs_0 = y.to(torch.complex128).T @ (filt * w)

    # For higher-order lifts we reuse `_stack_filtered`, which recursively
    # writes <D^k (Df - lam f), phi> = <Df - lam f, (-1)^k D^k phi>.
    U_lam = _stack_filtered(basis, ts, u, L, lam_c)
    Y_lam = _stack_filtered(basis, ts, y, K, lam_c)

    # The "order 0" block was already computed above; overwrite it for
    # consistency even though `_stack_filtered` does the same thing.
    m = u.shape[1]
    p = y.shape[1]
    U_lam[:m] = u_lam_pairs_0
    Y_lam[:p] = y_lam_pairs_0

    return torch.cat([Y_lam, U_lam], dim=0)  # (Kp + Lm, N)


def _trap_weights(ts: torch.Tensor) -> torch.Tensor:
    w = torch.zeros_like(ts)
    dts = ts[1:] - ts[:-1]
    w[:-1] += 0.5 * dts
    w[1:] += 0.5 * dts
    return w


def _stack_filtered(
    basis: TestFunctionBasis,
    ts: torch.Tensor,
    signal: torch.Tensor,
    orders: int,
    lam: complex,
) -> torch.Tensor:
    """Stack <D^k (Df - lam f), Phi> for k = 0..orders-1 via IBP."""
    d = signal.shape[1]
    # <D^k(Df - lam f), phi> = <Df - lam f, (-1)^k D^k phi>
    #                         = -<f, D(phi_k)> - lam <f, phi_k> with phi_k = (-1)^k D^k phi
    # Simpler: use the identity
    #   <D^{k+1} f, phi> = (-1)^{k+1} <f, D^{k+1} phi>,
    #   <D^k f,     phi> = (-1)^k     <f, D^k     phi>,
    # and linearity.
    w = _trap_weights(ts)  # (N_time,)
    lam_c = complex(lam)
    is_complex = abs(lam_c.imag) > 1e-15
    real_dtype = signal.dtype
    out_dtype = torch.complex128 if is_complex else real_dtype
    sig_cast = signal.to(out_dtype)
    w_cast = w.to(out_dtype)
    lam_scalar = lam_c if is_complex else float(lam_c.real)
    blocks = []
    for k in range(orders):
        phi_k = basis.evaluate(ts, order=k).to(out_dtype)
        phi_k1 = basis.evaluate(ts, order=k + 1).to(out_dtype)
        # <D^{k+1} f - lam D^k f, phi> = (-1)^{k+1} <f, phi^{(k+1)}> - lam (-1)^k <f, phi^{(k)}>.
        sgn_k = (-1.0) ** k
        sgn_kp1 = (-1.0) ** (k + 1)
        filt = sgn_kp1 * phi_k1 - lam_scalar * sgn_k * phi_k  # (N_time, N)
        block = sig_cast.T @ (filt * w_cast.unsqueeze(1))  # (d, N)
        blocks.append(block)
    return torch.cat(blocks, dim=0)  # (orders * d, N)


__all__ = [
    "LiftData",
    "compute_state_lifts",
    "compute_io_lifts",
    "compute_filtered_state_lift",
    "compute_filtered_io_lift",
]

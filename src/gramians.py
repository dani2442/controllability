"""Gramian computation for behavioral controllability analysis.

Implements the derivative-lifted Gramians G_{L,K}(λ) and K_{L,K}(λ)
from the continuous-time data-driven Hautus test.

Main functions:
    - compute_derivative_lift: Λ_L(u_λ, y_λ)
    - compute_G_LK: G_{L,K}(λ) Gramian
    - compute_K_LK: K_{L,K}(λ) matrix and eigenvalues
    - check_controllability: Test via rank of G_{L,K}(λ)
    - compute_persistent_excitation_gramian: Γ_L(u) Gramian
    - check_persistent_excitation: Persistent excitation test
"""

import torch
from typing import Tuple, Optional

from .utils import complex_dtype_from_real, to_complex


def compute_filtered_signal(
    signal: torch.Tensor,
    lam: complex,
    dt: float,
) -> torch.Tensor:
    """Compute λ-filtered signal: f_λ = df/dt - λf ≈ Δf/Δt - λf.
    
    Uses finite differences for derivative approximation.
    
    Args:
        signal: Input signal (N, d)
        lam: Complex filtering parameter λ
        dt: Time step
        
    Returns:
        f_λ: Filtered signal (N-1, d)
    """
    device = signal.device
    real_dtype = signal.dtype
    complex_dtype = complex_dtype_from_real(real_dtype)
    
    lam_c = torch.tensor(lam, device=device, dtype=complex_dtype)
    signal_c = to_complex(signal, complex_dtype)
    
    # Finite difference derivative: df/dt ≈ (f[k+1] - f[k]) / dt
    df = (signal_c[1:] - signal_c[:-1]) / dt  # (N-1, d)
    f0 = signal_c[:-1]  # (N-1, d)
    
    return df - lam_c * f0


def compute_derivative_lift(
    signal: torch.Tensor,
    L: int,
    dt: float,
) -> torch.Tensor:
    """Compute derivative lift Λ_L(f) = [f; df/dt; d²f/dt²; ...; d^{L-1}f/dt^{L-1}].
    
    Uses finite differences for derivative approximation.
    
    Args:
        signal: Input signal (N, d)
        L: Number of derivative levels (including 0th order)
        dt: Time step
        
    Returns:
        Lambda_L: Lifted signal (N-L+1, L*d)
    """
    device = signal.device
    dtype = signal.dtype
    N, d = signal.shape
    
    # Compute all derivatives
    derivatives = [signal]
    current = signal
    for _ in range(L - 1):
        # Finite difference derivative
        deriv = (current[1:] - current[:-1]) / dt
        derivatives.append(deriv)
        current = deriv
    
    # Truncate to common length
    min_len = N - L + 1
    truncated = [der[:min_len] for der in derivatives]
    
    # Stack: [f, df, d²f, ...]
    return torch.cat(truncated, dim=-1)  # (N-L+1, L*d)


def compute_persistent_excitation_gramian(
    u: torch.Tensor,
    order: int,
    dt: float,
) -> torch.Tensor:
    """Compute the persistent excitation Gramian Γ_order(u).

    Definition:
        Γ_L(u) = ∫_0^T Λ_L(u)(t) Λ_L(u)(t)^T dt

    with the derivative lift:
        Λ_L(u) = [u; du/dt; ...; d^{L-1}u/dt^{L-1}]

    This implementation uses finite differences for derivatives and a Riemann sum
    for the integral:
        Γ ≈ Σ_k Λ_k Λ_k^T Δt

    Args:
        u: Input trajectory (N, m)
        order: Excitation order L (number of derivative levels, including 0th)
        dt: Time step

    Returns:
        Gamma: Gramian matrix (order*m, order*m)
    """
    if order <= 0:
        raise ValueError(f"order must be positive (got {order}).")
    if u.ndim != 2:
        raise ValueError(f"u must have shape (N, m) (got {tuple(u.shape)}).")
    if u.shape[0] < order:
        raise ValueError(
            f"Need at least order samples to build Λ_order(u): "
            f"N={u.shape[0]} < order={order}."
        )

    Lambda = compute_derivative_lift(u, order, dt)  # (N-order+1, order*m)
    Gamma = Lambda.T @ Lambda * dt
    return 0.5 * (Gamma + Gamma.T)


def check_persistent_excitation(
    u: torch.Tensor,
    order: int,
    dt: float,
    threshold: float = 1e-8,
) -> dict:
    """Check whether u is persistently exciting of a given order.

    Uses a thresholded positive-definiteness test on Γ_order(u).

    Args:
        u: Input trajectory (N, m)
        order: Excitation order L
        dt: Time step
        threshold: Eigenvalue threshold for numerical positivity

    Returns:
        Dictionary with:
            - is_persistently_exciting: Boolean
            - gramian: Γ_order(u)
            - eigenvalues: Eigenvalues of Γ_order(u) (ascending)
            - min_eigenvalue: Smallest eigenvalue
            - rank: Thresholded rank
            - full_dimension: order*m
    """
    Gamma = compute_persistent_excitation_gramian(u, order, dt)
    eigvals = torch.linalg.eigvalsh(Gamma).real
    rank = (eigvals > threshold).sum().item()
    full_dimension = order * u.shape[1]
    return {
        "is_persistently_exciting": bool((eigvals.min() > threshold).item()),
        "gramian": Gamma,
        "eigenvalues": eigvals,
        "min_eigenvalue": eigvals.min().item(),
        "rank": rank,
        "full_dimension": full_dimension,
    }


def compute_filtered_derivative_lift(
    signal: torch.Tensor,
    L: int,
    lam: complex,
    dt: float,
) -> torch.Tensor:
    """Compute derivative lift of λ-filtered signal: Λ_L(f_λ).
    
    First filters: f_λ = df/dt - λf
    Then lifts: Λ_L(f_λ) = [f_λ; df_λ/dt; ...; d^{L-1}f_λ/dt^{L-1}]
    
    Args:
        signal: Input signal (N, d)
        L: Number of derivative levels
        lam: Complex filtering parameter λ
        dt: Time step
        
    Returns:
        Lambda_L_filtered: Lifted filtered signal (N-L, L*d)
    """
    # First apply λ-filter
    f_lam = compute_filtered_signal(signal, lam, dt)  # (N-1, d)
    
    # Then apply derivative lift
    return compute_derivative_lift(f_lam, L, dt)  # (N-L, L*d)


def compute_G_LK(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
    dt: float,
) -> torch.Tensor:
    """Compute the derivative-lifted Gramian G_{L,K}(λ).
    
    G_{L,K}(λ) = ∫ Λ_{L,K}(u_λ, y_λ)(t) Λ_{L,K}(u_λ, y_λ)(t)^* dt
    
    where u_λ = du/dt - λu, y_λ = dy/dt - λy, and
    Λ_{L,K}(u_λ, y_λ) = [Λ_L(u_λ); Λ_K(y_λ)]
    
    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        lam: Complex parameter λ
        dt: Time step
        
    Returns:
        G_LK: Gramian matrix (Lm + Kp, Lm + Kp)
    """
    device = u.device
    complex_dtype = complex_dtype_from_real(u.dtype)
    
    m = u.shape[1]
    p = y.shape[1]
    
    # Compute filtered derivative lifts
    Lambda_L_u = compute_filtered_derivative_lift(u, L, lam, dt)  # (N', Lm)
    Lambda_K_y = compute_filtered_derivative_lift(y, K, lam, dt)  # (N'', Kp)
    
    # Align lengths (take minimum)
    N_common = min(Lambda_L_u.shape[0], Lambda_K_y.shape[0])
    Lambda_L_u = Lambda_L_u[:N_common]
    Lambda_K_y = Lambda_K_y[:N_common]
    
    # Stack: Λ_{L,K}(u_λ, y_λ) = [Λ_L(u_λ); Λ_K(y_λ)]
    Lambda_LK = torch.cat([Lambda_L_u, Lambda_K_y], dim=-1)  # (N', Lm + Kp)
    Lambda_LK = to_complex(Lambda_LK, complex_dtype)
    
    # Gramian: G = ∫ Λ Λ^* dt ≈ Σ Λ_k Λ_k^* Δt
    G = Lambda_LK.conj().T @ Lambda_LK * dt  # (Lm + Kp, Lm + Kp)
    
    return G


def compute_K_LK(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
    dt: float,
    *,
    expected_rank: Optional[int] = None,
    basis_method: str = "svd",
    rank_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K_{L,K}(λ) matrix and its eigenvalues.
    
    K_{L,K}(λ) is related to the cross-moment matrix that determines
    the finite candidate set for controllability checking.
    
    Following Theorem 4 (finite candidate set), define:
        Ξ(t) := Λ_{L,K}(u,y)(t) ∈ R^{Lm+Kp}
        S := Γ(Ξ) ≈ (√dt Ξ^T) ∈ R^{(Lm+Kp)×q}
    Let U have orthonormal columns spanning im S (rank r = Lm+n). Then with
        ξ(t) := U^* Ξ(t) ∈ C^r,
    we compute the reduced matrix
        K = (∫ ξ ξ^* dt)^{-1} ∫ ξ \\dot{ξ}^* dt,
    using finite differences and a Riemann sum.
    
    The eigenvalues of K form the candidate set for controllability testing.
    
    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        lam: Complex parameter λ (unused; included for API compatibility)
        dt: Time step
        expected_rank: If provided, use this rank r for the subspace basis U.
        basis_method: How to compute U. One of {"svd", "qr"}.
        rank_tol: Absolute tolerance on singular values for rank detection when
            expected_rank is None.
        
    Returns:
        K_matrix: The K_{L,K}(λ) matrix
        eigenvalues: Complex eigenvalues of K
    """
    complex_dtype = complex_dtype_from_real(u.dtype)

    # Build Ξ(t) = Λ_{L,K}(u,y) from (unfiltered) derivative lifts.
    Lambda_L_u = compute_derivative_lift(u, L, dt)  # (N_u, Lm)
    Lambda_K_y = compute_derivative_lift(y, K, dt)  # (N_y, Kp)

    N_common = min(Lambda_L_u.shape[0], Lambda_K_y.shape[0])
    if N_common < 2:
        raise ValueError(
            f"Need at least 2 aligned samples to compute K (got N_common={N_common})."
        )

    Xi = torch.cat([Lambda_L_u[:N_common], Lambda_K_y[:N_common]], dim=-1)  # (q, d)
    Xi = to_complex(Xi, complex_dtype)

    # Weighted sample matrix S = √dt Ξ^T; im(S) = im(Γ(Ξ)).
    Xi0 = Xi[:-1]  # (q-1, d)
    sqrt_dt = torch.sqrt(
        torch.tensor(float(dt), device=Xi0.device, dtype=Xi0.real.dtype)
    )
    S = (Xi0.T * sqrt_dt).to(complex_dtype)  # (d, q-1)

    def _compute_subspace_basis(
        S_factor: torch.Tensor,
        *,
        expected_rank: Optional[int],
        method: str,
        rank_tol: Optional[float],
    ) -> torch.Tensor:
        d, q = S_factor.shape
        if method not in {"svd", "qr"}:
            raise ValueError(
                f"basis_method must be one of {{'svd','qr'}} (got {method})."
            )

        def _default_tol(smax: torch.Tensor) -> torch.Tensor:
            eps = torch.finfo(smax.dtype).eps
            return torch.as_tensor(max(d, q), device=smax.device, dtype=smax.dtype) * eps * smax

        if expected_rank is not None:
            if expected_rank <= 0:
                raise ValueError(f"expected_rank must be positive (got {expected_rank}).")
            r = int(expected_rank)
            if r > d:
                raise ValueError(
                    f"expected_rank={r} exceeds ambient dimension d={d}."
                )
        else:
            if method == "qr":
                # Not rank-revealing without pivoting; use diag(R) as a heuristic.
                _, R = torch.linalg.qr(S_factor, mode="reduced")
                diag = torch.abs(torch.diagonal(R[: R.shape[0], : R.shape[0]]))
                smax = (
                    diag.max()
                    if diag.numel()
                    else torch.tensor(0.0, device=diag.device, dtype=diag.dtype)
                )
                tol = _default_tol(smax) if rank_tol is None else torch.as_tensor(
                    rank_tol, device=diag.device, dtype=diag.dtype
                )
                r = int((diag > tol).sum().item())
            else:
                # "SVD" via Gramian eigendecomposition (avoids forming Vh of shape (d, q)).
                gram = S_factor @ S_factor.conj().T
                gram = 0.5 * (gram + gram.conj().T)
                evals = torch.linalg.eigvalsh(gram).real
                svals = torch.sqrt(torch.clamp(evals, min=0.0))
                smax = (
                    svals.max()
                    if svals.numel()
                    else torch.tensor(0.0, device=svals.device, dtype=svals.dtype)
                )
                tol = _default_tol(smax) if rank_tol is None else torch.as_tensor(
                    rank_tol, device=svals.device, dtype=svals.dtype
                )
                r = int((svals > tol).sum().item())

        if r <= 0:
            raise ValueError(
                "Estimated rank is 0; try increasing data length, reducing noise, "
                "or provide expected_rank explicitly."
            )

        if method == "qr":
            Q, _ = torch.linalg.qr(S_factor, mode="reduced")  # Q: (d, k)
            if r > Q.shape[1]:
                raise ValueError(
                    f"expected_rank={r} exceeds QR basis size {Q.shape[1]} "
                    f"(need at least q >= r; got q={q})."
                )
            return Q[:, :r]

        gram = S_factor @ S_factor.conj().T
        gram = 0.5 * (gram + gram.conj().T)
        evals, evecs = torch.linalg.eigh(gram)  # ascending
        idx = torch.argsort(evals.real, descending=True)
        evecs = evecs[:, idx]
        return evecs[:, :r]

    U = _compute_subspace_basis(
        S,
        expected_rank=expected_rank,
        method=basis_method,
        rank_tol=rank_tol,
    )  # (d, r)

    # Reduced coordinates ξ(t) = U^* Ξ(t).
    xi = Xi @ U  # (q, r)
    xi0 = xi[:-1]  # (q-1, r)
    dxi = xi[1:] - xi[:-1]  # (q-1, r), no /dt since it cancels

    G = xi0.conj().T @ xi0 * dt
    M = xi0.conj().T @ dxi

    try:
        K_matrix = torch.linalg.solve(G, M)
    except RuntimeError:
        # Fall back to least-squares if G is singular/ill-conditioned.
        K_matrix = torch.linalg.lstsq(G, M).solution

    eigenvalues = torch.linalg.eigvals(K_matrix)
    return K_matrix, eigenvalues


def check_controllability(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    m: int,
    dt: float,
    candidate_lambdas: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
) -> dict:
    """Check controllability via rank of G_{L,K}(λ).
    
    Behavior is controllable iff rank(G_{L,K}(λ)) = Lm + n for all λ ∈ C.
    
    Uses thresholded rank: count eigenvalues > threshold.
    
    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        n: State dimension
        m: Input dimension
        dt: Time step
        candidate_lambdas: Candidate λ values to check (if None, computed from data)
        threshold: Eigenvalue threshold for rank computation
        
    Returns:
        Dictionary with:
            - is_controllable: Boolean
            - expected_rank: Lm + n
            - ranks: Ranks at each candidate λ
            - min_eigenvalues: Smallest eigenvalue at each λ
            - candidate_lambdas: The λ values tested
    """
    device = u.device
    
    expected_rank = L * m + n
    
    # Get candidate eigenvalues if not provided
    if candidate_lambdas is None:
        _, candidate_lambdas = compute_K_LK(
            u, y, L, K, 0.0, dt, expected_rank=expected_rank
        )
    
    ranks = []
    min_eigs = []
    
    for lam in candidate_lambdas:
        lam_val = lam.item() if torch.is_tensor(lam) else lam
        
        G = compute_G_LK(u, y, L, K, lam_val, dt)
        
        # Eigenvalues of Hermitian matrix
        eigvals = torch.linalg.eigvalsh(G).real
        eigvals_sorted = torch.sort(eigvals, descending=True).values
        
        # Thresholded rank
        rank = (eigvals > threshold).sum().item()
        ranks.append(rank)
        
        # Track the (Lm+n)-th eigenvalue (controllability margin)
        if len(eigvals_sorted) >= expected_rank:
            min_eigs.append(eigvals_sorted[expected_rank - 1].item())
        else:
            min_eigs.append(0.0)
    
    ranks = torch.tensor(ranks)
    min_eigs = torch.tensor(min_eigs)
    
    # Controllable if all ranks equal expected
    is_controllable = (ranks >= expected_rank).all().item()
    
    return {
        "is_controllable": is_controllable,
        "expected_rank": expected_rank,
        "ranks": ranks,
        "min_eigenvalues": min_eigs,
        "candidate_lambdas": candidate_lambdas,
    }

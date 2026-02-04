"""Gramian computation for behavioral controllability analysis.

Implements the derivative-lifted Gramians G_{L,K}(λ) and K_{L,K}(λ)
from the continuous-time data-driven Hautus test.

Main functions:
    - compute_derivative_lift: Λ_L(u_λ, y_λ)
    - compute_G_LK: G_{L,K}(λ) Gramian
    - compute_K_LK: K_{L,K}(λ) matrix and eigenvalues
    - check_controllability: Test via rank of G_{L,K}(λ)
"""

import torch
from typing import Tuple, Union, Optional

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K_{L,K}(λ) matrix and its eigenvalues.
    
    K_{L,K}(λ) is related to the cross-moment matrix that determines
    the finite candidate set for controllability checking.
    
    K = (∫ Λ Λ^* dt)^{-1} (∫ Λ dΛ^* dt)
    
    The eigenvalues of K form the candidate set for controllability testing.
    
    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        lam: Complex parameter λ
        dt: Time step
        
    Returns:
        K_matrix: The K_{L,K}(λ) matrix
        eigenvalues: Complex eigenvalues of K
    """
    device = u.device
    complex_dtype = complex_dtype_from_real(u.dtype)
    
    # Compute derivative lifts (unfiltered, for cross-moment)
    max_deriv = max(L, K)
    Lambda_L_u = compute_derivative_lift(u, L, dt)  # (N', Lm)
    Lambda_K_y = compute_derivative_lift(y, K, dt)  # (N'', Kp)
    
    # Align lengths
    N_common = min(Lambda_L_u.shape[0], Lambda_K_y.shape[0]) - 1
    Lambda_L_u = Lambda_L_u[:N_common]
    Lambda_K_y = Lambda_K_y[:N_common]
    
    # Stack
    Lambda = torch.cat([Lambda_L_u, Lambda_K_y], dim=-1)  # (N', d)
    Lambda = to_complex(Lambda, complex_dtype)
    
    # Derivative of Lambda
    dLambda = (Lambda[1:] - Lambda[:-1])  # (N'-1, d), no /dt since it cancels
    Lambda0 = Lambda[:-1]  # (N'-1, d)
    
    # Gramians
    G = Lambda0.conj().T @ Lambda0 * dt  # ∫ Λ Λ^* dt
    M = dLambda.T @ Lambda0.conj()  # ∫ dΛ Λ^* (using Δ instead of dt)
    
    # K = G^{-1} M^T
    K_matrix = torch.linalg.solve(G, M.T)
    
    # Eigenvalues
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
        _, candidate_lambdas = compute_K_LK(u, y, L, K, 0.0, dt)
    
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

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
    T: float,
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
        T: Total time (used for normalization)
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
    
    return G/T


def compute_K_LK(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    lam: complex,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K_{L,K}(λ) matrix and its eigenvalues.
    
    K_{L,K}(λ) is related to the cross-moment matrix that determines
    the finite candidate set for controllability checking.
    
    K = (∫ ξ ξ^* dt)^{-1} (∫ ξ dξ^* dt), with ξ = U^T Λ and
    U the first r = Lm + n left singular vectors of S = Γ(Λ).
    
    The eigenvalues of K form the candidate set for controllability testing.
    
    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        n: State dimension (used for r = Lm + n)
        lam: Complex parameter λ (unused; kept for API compatibility)
        dt: Time step
        
    Returns:
        K_matrix: The K_{L,K}(λ) matrix
        eigenvalues: Complex eigenvalues of K
    """
    complex_dtype = complex_dtype_from_real(u.dtype)
    _ = lam  # unused; kept for API compatibility
    
    # Compute derivative lifts (unfiltered)
    Lambda_L_u = compute_derivative_lift(u, L, dt)  # (N', Lm)
    Lambda_K_y = compute_derivative_lift(y, K, dt)  # (N'', Kp)
    
    # Align lengths
    N_common = min(Lambda_L_u.shape[0], Lambda_K_y.shape[0])
    if N_common < 2:
        raise ValueError("Not enough samples to build K_{L,K}: need at least 2.")
    Lambda_L_u = Lambda_L_u[:N_common]
    Lambda_K_y = Lambda_K_y[:N_common]
    
    # Stack
    Lambda = torch.cat([Lambda_L_u, Lambda_K_y], dim=-1)  # (N, d)
    
    # Build S = Γ(Λ) and compute U via SVD
    S = Lambda.T  # (d, N)
    U_hat, _, _ = torch.linalg.svd(S, full_matrices=False)
    m = u.shape[1]
    r = L * m + n
    if r > U_hat.shape[1]:
        raise ValueError(
            f"Requested r = Lm + n = {r}, but only {U_hat.shape[1]} singular vectors "
            "are available. Provide more data or reduce r."
        )
    U = U_hat[:, :r]  # (d, r)
    
    # Reduced signal ξ = U^T Λ
    xi = Lambda @ U  # (N, r)
    xi = to_complex(xi, complex_dtype)
    
    # Derivative of ξ
    dxi = xi[1:] - xi[:-1]  # (N-1, r), no /dt since it cancels
    xi0 = xi[:-1]  # (N-1, r)
    
    # Gramians in reduced coordinates
    G = xi0.conj().T @ xi0 * dt  # ∫ ξ ξ^* dt
    M = dxi.T @ xi0.conj()  # ∫ dξ ξ^* (using Δ instead of dt)
    
    # K = G^{-1} M^T
    K_matrix = torch.linalg.lstsq(G, M.T).solution
    
    # Eigenvalues
    eigenvalues = torch.linalg.eigvals(K_matrix)
    
    return K_matrix, eigenvalues


def compute_H_LK(
    y: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
    T: float,
    dt: float,
) -> torch.Tensor:
    """Compute the reduced data matrix H_{L,K}(λ) = Γ_K(y_λ).

    Uses only the output trajectory (no input required).

    H_{L,K}(λ) = ∫ Λ_K(y_λ)(t) Λ_K(y_λ)(t)^* dt

    where y_λ = dy/dt − λy.

    Reference: Theorem thm:ct-dd-hautus-reduced.

    Args:
        y: Output trajectory (N, p)
        L: Number of input derivative levels (unused; included for naming consistency)
        K: Number of output derivative levels
        lam: Complex parameter λ
        T: Total time (used for normalization)
        dt: Time step

    Returns:
        H_LK: Gramian matrix (Kp, Kp)
    """
    complex_dtype = complex_dtype_from_real(y.dtype)

    # Λ_K(y_λ): first filter, then derivative-lift
    Lambda_K_y = compute_filtered_derivative_lift(y, K, lam, dt)  # (N', Kp)
    Lambda_K_y = to_complex(Lambda_K_y, complex_dtype)

    # H = ∫ Λ Λ^* dt ≈ Σ Λ_k Λ_k^* Δt
    H = Lambda_K_y.conj().T @ Lambda_K_y * dt  # (Kp, Kp)
    return H/T


def _compute_output_lift_svd(
    y: torch.Tensor,
    K: int,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SVD of the output derivative lift Υ = Λ_K(y).

    Returns:
        Upsilon: Lifted output signal (N', Kp)
        svals: Singular values of Upsilon (descending)
        Vh: Right singular vectors (min(N',Kp), Kp)
    """
    Upsilon = compute_derivative_lift(y, K, dt)  # (N', Kp)
    if Upsilon.shape[0] < 2:
        raise ValueError(
            "Not enough samples to build K^y_{L,K}: need at least 2."
        )
    _, svals, Vh = torch.linalg.svd(Upsilon, full_matrices=False)
    return Upsilon, svals, Vh


def compute_K_LK_reduced(
    y: torch.Tensor,
    L: int,
    K: int,
    dt: float,
    rank_tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K^y_{L,K} matrix and its eigenvalues (reduced / output-only).

    Implements Theorem thm:ct-dd-hautus-reduced-finite-lambda:

        Υ  = Λ_K(y)                         (derivative lift of y)
        S_y = Γ(Υ)  → SVD  → U_y (r = rank(S_y) right singular vectors)
        υ  = U_y^T Υ                        (reduced signal)
        K^y = (∫ υ υ^T dt)^{-1} ∫ υ υ̇^T dt

    The eigenvalues σ(K^y_{L,K}) form the finite candidate set: rank
    failure of H_{L,K}(λ) can occur only at λ ∈ σ(K^y_{L,K}).

    Args:
        y: Output trajectory (N, p)
        L: Number of input derivative levels (unused; included for naming consistency)
        K: Number of output derivative levels
        dt: Time step
        rank_tol: Relative tolerance for numerical rank of S_y

    Returns:
        K_matrix: The K^y_{L,K} matrix (r, r), r = rank(S_y)
        eigenvalues: Complex eigenvalues of K^y_{L,K}
    """
    complex_dtype = complex_dtype_from_real(y.dtype)

    # 1. Derivative lift of y: Υ = Λ_K(y)
    Upsilon, svals, Vh = _compute_output_lift_svd(y, K, dt)
    if svals.numel() == 0:
        raise ValueError("SVD failed: no singular values returned.")
    smax = svals.max()
    if smax == 0:
        raise ValueError("rank(S_y) = 0; output data is identically zero.")
    tol = rank_tol * smax
    r = int((svals > tol).sum().item())
    if r < 1:
        raise ValueError(
            f"rank(S_y) < 1 (tol={tol:.3e}). "
            "Increase data richness or reduce K."
        )
    # 2. U_y: orthonormal basis for im(S_y) in R^{Kp}
    U_y = Vh.conj().T[:, :r]  # (Kp, r)

    # 3. Reduced signal: υ = U_y^T Υ  (equivalently Υ @ U_y)
    upsilon = Upsilon @ U_y  # (N', r)
    upsilon = to_complex(upsilon, complex_dtype)

    # 4. Build K^y = (∫ υ υ^* dt)^{-1} · ∫ υ υ̇^* dt
    d_upsilon = upsilon[1:] - upsilon[:-1]  # Δυ (no /dt; cancels)
    upsilon0 = upsilon[:-1]

    G = upsilon0.conj().T @ upsilon0 * dt   # ∫ υ υ^* dt
    M = d_upsilon.T @ upsilon0.conj()        # Σ Δυ_k υ_k^*

    # Use lstsq for robustness when G is ill-conditioned
    K_matrix = torch.linalg.lstsq(G, M.T).solution  # G^{-1} M^T
    eigenvalues = torch.linalg.eigvals(K_matrix)

    return K_matrix, eigenvalues


def check_controllability_reduced(
    y: torch.Tensor,
    L: int,
    K: int,
    T: float,
    dt: float,
    candidate_lambdas: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    rank_tol: float = 1e-8,
) -> dict:
    """Check controllability via rank of H_{L,K}(λ) (reduced / output-only).

    Behaviour is controllable if rank(H_{L,K}(λ)) = Kp for all λ ∈ ℂ.
    The finite candidate set from K^y_{L,K} reduces this to finitely many λ.
    If rank(S_y) < Kp, the finite-candidate reduction is not conclusive.

    Args:
        y: Output trajectory (N, p)
        L: Number of input derivative levels (unused in H; kept for naming consistency)
        K: Number of output derivative levels
        T: Total time (used for normalization)
        dt: Time step
        candidate_lambdas: Candidate λ values (if None, computed from data)
        threshold: Eigenvalue threshold for rank computation
        rank_tol: Relative tolerance for numerical rank of S_y

    Returns:
        Dictionary with:
            - is_controllable: Boolean
            - expected_rank: Kp
            - ranks: Ranks at each candidate λ
            - min_eigenvalues: Smallest relevant eigenvalue at each λ
            - candidate_lambdas: The λ values tested
            - rank_Sy: rank of S_y = Γ(Λ_K(y))
            - full_rank_Sy: whether rank_Sy == Kp
    """
    p = y.shape[1]
    expected_rank = K * p

    # Rank of S_y = Γ(Λ_K(y)) for applicability of finite candidate set
    _, svals, _ = _compute_output_lift_svd(y, K, dt)
    if svals.numel() == 0:
        rank_Sy = 0
    else:
        tol = rank_tol * svals.max()
        rank_Sy = int((svals > tol).sum().item())
    full_rank_Sy = (rank_Sy == expected_rank)

    # Get candidate eigenvalues if not provided
    if candidate_lambdas is None:
        _, candidate_lambdas = compute_K_LK_reduced(
            y, L, K, dt, rank_tol=rank_tol
        )

    ranks = []
    min_eigs = []

    for lam in candidate_lambdas:
        lam_val = lam.item() if torch.is_tensor(lam) else lam

        H = compute_H_LK(y, L, K, lam_val, T, dt)

        # Eigenvalues of Hermitian matrix
        eigvals = torch.linalg.eigvalsh(H).real
        eigvals_sorted = torch.sort(eigvals, descending=True).values

        # Thresholded rank
        rank = (eigvals > threshold).sum().item()
        ranks.append(rank)

        # Track the Kp-th eigenvalue (controllability margin)
        if len(eigvals_sorted) >= expected_rank:
            min_eigs.append(eigvals_sorted[expected_rank - 1].item())
        else:
            min_eigs.append(0.0)

    ranks = torch.tensor(ranks)
    min_eigs = torch.tensor(min_eigs)

    is_controllable = bool(full_rank_Sy) and (ranks >= expected_rank).all().item()

    return {
        "is_controllable": is_controllable,
        "expected_rank": expected_rank,
        "ranks": ranks,
        "min_eigenvalues": min_eigs,
        "candidate_lambdas": candidate_lambdas,
        "rank_Sy": rank_Sy,
        "full_rank_Sy": full_rank_Sy,
    }


def check_controllability(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    m: int,
    T: float,
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
        T: Total time (used for normalization)
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
        _, candidate_lambdas = compute_K_LK(u, y, L, K, n, 0.0, dt)
    
    ranks = []
    min_eigs = []
    
    for lam in candidate_lambdas:
        lam_val = lam.item() if torch.is_tensor(lam) else lam
        
        G = compute_G_LK(u, y, L, K, lam_val, T, dt)
        
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

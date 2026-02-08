"""Gramian computation for behavioral controllability analysis.

Implements derivative-lifted matrices from continuous-time data-driven
Hautus tests:
    - full test: G_{L,K}(λ)
    - reduced output-only test: H_{L,K}(λ), K^y_{L,K}
    - observable-quotient test: Q_{L,K}(λ)
"""

import torch
from typing import Tuple, Union, Optional, Dict, Any

from .utils import (
    complex_dtype_from_real,
    compute_lift_matrix,
    compute_observability_matrix,
    compute_toeplitz_matrix,
    to_complex,
)


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
    
    G_{L,K}(λ) = 1/T*∫ Λ_{L,K}(u_λ, y_λ)(t) Λ_{L,K}(u_λ, y_λ)(t)^* dt
    
    where u_λ = du/dt - λu, y_λ = dy/dt - λy, and
    Λ_{L,K}(u_λ, y_λ) = [Λ_L(u_λ); Λ_K(y_λ)]

    Row-stacked convention:
        If Z stores samples as rows z_k^T, then Γ(z) ≈ dt * Z^T Z^*.
    
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
    complex_dtype = complex_dtype_from_real(u.dtype)
    
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
    G = Lambda_LK.T @ Lambda_LK.conj() * dt  # (Lm + Kp, Lm + Kp)
    
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
    
    K = (∫ ξ ξ^* dt)^{-1} (∫ ξ dξ^*), with ξ = Λ U and
    U the first r = Lm + n right singular vectors of Λ.
    
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
    
    # Informative coordinates are given by right singular vectors of Λ.
    _, _, Vh = torch.linalg.svd(Lambda, full_matrices=False)
    m = u.shape[1]
    r = L * m + n
    if r > Vh.shape[0]:
        raise ValueError(
            f"Requested r = Lm + n = {r}, but only {Vh.shape[0]} singular vectors "
            "are available. Provide more data or reduce r."
        )
    U = Vh.conj().T[:, :r]  # (d, r)
    
    # Reduced signal ξ = Λ U (row-stacked samples)
    xi = Lambda @ U  # (N, r)
    xi = to_complex(xi, complex_dtype)
    
    # Derivative of ξ
    dxi = xi[1:] - xi[:-1]  # (N-1, r), no /dt since it cancels
    xi0 = xi[:-1]  # (N-1, r)
    
    # Moments in reduced coordinates under row-stacked convention.
    G = xi0.T @ xi0.conj() * dt  # ∫ ξ ξ^* dt
    G = 0.5 * (G + G.conj().T)
    M = xi0.T @ dxi.conj()       # ∫ ξ dξ^* (dt cancels with finite difference)

    # K = G^{-1} M
    K_matrix = torch.linalg.lstsq(G, M).solution
    
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

    Row-stacked convention:
        If Z stores samples as rows z_k^T, then Γ(z) ≈ dt * Z^T Z^*.

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
    H = Lambda_K_y.T @ Lambda_K_y.conj() * dt  # (Kp, Kp)
    return H/T


def compute_observable_quotient_coordinates(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    dt: float,
    rank_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Build observable quotient coordinates from input-output data.

    Constructs:
        Xi        = Λ_{L,K}(u, y)
        U         = right singular vectors spanning rank-r informative subspace
        xi        = Xi U
        Gamma_xi  = ∫ xi xi^* dt

    Row-stacked convention:
        If X stores samples as rows x_k^T, then Γ(x) ≈ dt * X^T X^*.

    with target rank r = Lm + n.

    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        n: State dimension for target rank r = Lm + n
        dt: Time step
        rank_tol: Relative tolerance for numerical rank based on singular values

    Returns:
        Dictionary with:
            - Xi: Complex lifted signal (N_common, Lm+Kp)
            - U: Basis matrix (Lm+Kp, r)
            - xi: Reduced coordinates (N_common, r)
            - Gamma_xi: Gramian ∫ xi xi^* dt, shape (r, r)
            - target_rank: r
            - numerical_rank: numerical rank of Xi
            - condition_number: cond(Gamma_xi)
            - N_common: number of aligned samples
    """
    if u.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"u and y must both have shape (N, d). Got u={tuple(u.shape)}, y={tuple(y.shape)}."
        )
    if u.shape[0] != y.shape[0]:
        raise ValueError(
            f"u and y must have the same number of samples. Got {u.shape[0]} and {y.shape[0]}."
        )
    if L <= 0 or K <= 0:
        raise ValueError(f"L and K must be positive. Got L={L}, K={K}.")

    m = u.shape[1]
    target_rank = L * m + n
    complex_dtype = complex_dtype_from_real(u.dtype)

    lambda_L_u = compute_derivative_lift(u, L, dt)
    lambda_K_y = compute_derivative_lift(y, K, dt)
    n_common = min(lambda_L_u.shape[0], lambda_K_y.shape[0])
    if n_common < 2:
        raise ValueError(
            "Not enough aligned samples to build observable quotient coordinates "
            f"(N_common={n_common})."
        )

    Xi_real = torch.cat([lambda_L_u[:n_common], lambda_K_y[:n_common]], dim=-1)
    Xi = to_complex(Xi_real, complex_dtype)

    # Xi = U_svd S V^*, informative coordinates are in the right singular vectors V.
    _, svals, Vh = torch.linalg.svd(Xi, full_matrices=False)
    if svals.numel() == 0:
        raise ValueError("SVD of Xi returned no singular values.")
    smax = svals.max().real
    if smax <= 0:
        raise ValueError("Xi is numerically zero; cannot build informative coordinates.")
    tol = rank_tol * smax
    numerical_rank = int((svals.real > tol).sum().item())

    max_available_rank = Vh.shape[0]
    if target_rank > max_available_rank:
        raise ValueError(
            f"Target rank r=Lm+n={target_rank} exceeds available SVD rank basis "
            f"{max_available_rank}. Provide more data or reduce L."
        )
    if numerical_rank < target_rank:
        raise ValueError(
            f"Insufficient numerical rank for observable quotient coordinates: "
            f"rank(Xi)={numerical_rank} < r=Lm+n={target_rank} "
            f"(tol={float(tol):.3e})."
        )

    U = Vh.conj().T[:, :target_rank]   # (Lm+Kp, r)
    xi = Xi @ U                         # (N_common, r)
    gamma_xi = xi.T @ xi.conj() * dt    # ∫ xi xi^* dt
    gamma_xi = 0.5 * (gamma_xi + gamma_xi.conj().T)

    eigvals = torch.linalg.eigvalsh(gamma_xi).real
    min_eig = float(eigvals.min().item())
    max_eig = float(eigvals.max().item())
    cond = float("inf") if min_eig <= 0.0 else max_eig / min_eig

    return {
        "Xi": Xi,
        "U": U,
        "xi": xi,
        "Gamma_xi": gamma_xi,
        "target_rank": target_rank,
        "numerical_rank": numerical_rank,
        "condition_number": cond,
        "N_common": n_common,
    }


def compute_Q_LK_from_coordinates(
    y: torch.Tensor,
    K: int,
    lam: complex,
    dt: float,
    xi: torch.Tensor,
    Gamma_xi: torch.Tensor,
) -> torch.Tensor:
    """Compute Q_{L,K}(λ) from precomputed observable quotient coordinates.

    Q_{L,K}(λ) = (∫ Λ_K(y_λ) xi^* dt) (∫ xi xi^* dt)^{-1}.

    Row-stacked convention:
        Γ(xi) ≈ dt * Xi^T Xi^*, and cross ≈ dt * Y^T Xi^*.

    Args:
        y: Output trajectory (N, p)
        K: Number of output derivative levels
        lam: Complex parameter λ
        dt: Time step
        xi: Reduced coordinates from compute_observable_quotient_coordinates
        Gamma_xi: Gramian ∫ xi xi^* dt from compute_observable_quotient_coordinates

    Returns:
        Q: Quotient matrix with shape (Kp, r), r = xi.shape[1]
    """
    if xi.ndim != 2:
        raise ValueError(f"xi must have shape (N, r). Got {tuple(xi.shape)}.")
    if Gamma_xi.ndim != 2 or Gamma_xi.shape[0] != Gamma_xi.shape[1]:
        raise ValueError(
            f"Gamma_xi must be square. Got shape {tuple(Gamma_xi.shape)}."
        )
    if Gamma_xi.shape[0] != xi.shape[1]:
        raise ValueError(
            f"Incompatible xi and Gamma_xi: xi has width {xi.shape[1]}, "
            f"Gamma_xi is {tuple(Gamma_xi.shape)}."
        )

    complex_dtype = xi.dtype if torch.is_complex(xi) else complex_dtype_from_real(y.dtype)
    xi_c = to_complex(xi, complex_dtype)
    gamma_c = to_complex(Gamma_xi, complex_dtype)

    lambda_K_y_lam = compute_filtered_derivative_lift(y, K, lam, dt)
    lambda_K_y_lam = to_complex(lambda_K_y_lam, complex_dtype)

    n_common = min(lambda_K_y_lam.shape[0], xi_c.shape[0])
    if n_common < 1:
        raise ValueError(
            "Not enough aligned samples between Λ_K(y_λ) and xi "
            f"(N_common={n_common})."
        )

    y_lam = lambda_K_y_lam[:n_common]
    xi_used = xi_c[:n_common]
    # If alignment truncated xi, use the aligned Gramian to preserve consistency.
    if n_common == xi_c.shape[0]:
        gamma_used = gamma_c
    else:
        gamma_used = xi_used.T @ xi_used.conj() * dt
        gamma_used = 0.5 * (gamma_used + gamma_used.conj().T)

    cross = y_lam.T @ xi_used.conj() * dt  # (Kp, r): ∫ Λ_K(y_λ) ξ^* dt

    # Solve Q * Gamma = Cross by transposing into a left solve.
    try:
        Q = torch.linalg.solve(gamma_used, cross.T).T
    except RuntimeError:
        Q = torch.linalg.lstsq(gamma_used, cross.T).solution.T

    return Q


def compute_Q_LK(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    lam: complex,
    dt: float,
    rank_tol: float = 1e-8,
    return_aux: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """One-shot wrapper to compute Q_{L,K}(λ) from (u, y) data.

    Args:
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        L: Number of input derivative levels
        K: Number of output derivative levels
        n: State dimension for target rank r = Lm + n
        lam: Complex parameter λ
        dt: Time step
        rank_tol: Relative tolerance for coordinate-rank checks
        return_aux: If True, also return coordinate diagnostics

    Returns:
        Q if return_aux=False, else (Q, aux_dict)
    """
    aux = compute_observable_quotient_coordinates(
        u=u,
        y=y,
        L=L,
        K=K,
        n=n,
        dt=dt,
        rank_tol=rank_tol,
    )
    Q = compute_Q_LK_from_coordinates(
        y=y,
        K=K,
        lam=lam,
        dt=dt,
        xi=aux["xi"],
        Gamma_xi=aux["Gamma_xi"],
    )
    if not return_aux:
        return Q
    aux_out = dict(aux)
    aux_out["lambda"] = lam
    return Q, aux_out


def compute_model_basis_R(
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Compute model-only canonical basis and coordinate matrix.

    Given M_{L,K}(A,B,C,D), define:
        G_M := M^* M,
        U_mod := M G_M^{-1/2},
        R_mod := U_mod^* M = G_M^{1/2}.

    Args:
        C, A, B, D: System matrices.
        L: Input derivative levels.
        K: Output derivative levels.
        eps: Relative eigenvalue floor used for stable inverse-square-root.

    Returns:
        Dictionary with keys:
            - M: Lift matrix M_{L,K}
            - G_M: Hermitian Gram matrix M^* M
            - U_model: Canonical orthonormal basis of im(M)
            - R_model: Canonical coordinate matrix G_M^{1/2}
            - inv_sqrt_G_M: G_M^{-1/2}
            - eigvals_clamped: clamped eigenvalues used in decomposition
            - ortho_residual: ||U_model^*U_model - I||_2
            - factor_residual: ||U_model^*M - R_model||_2
    """
    if L <= 0 or K <= 0:
        raise ValueError(f"L and K must be positive. Got L={L}, K={K}.")

    M_real = compute_lift_matrix(C=C, A=A, B=B, D=D, L=L, K=K)
    if torch.is_complex(M_real):
        work_dtype = M_real.dtype
    else:
        work_dtype = complex_dtype_from_real(M_real.dtype)
    M = to_complex(M_real, work_dtype)

    G_M = M.conj().T @ M
    G_M = 0.5 * (G_M + G_M.conj().T)

    eigvals, eigvecs = torch.linalg.eigh(G_M)
    eigvals_r = eigvals.real
    max_eval = float(torch.max(torch.abs(eigvals_r)).item()) if eigvals_r.numel() > 0 else 0.0
    floor = max(eps * max(max_eval, 1.0), eps)
    eigvals_clamped = torch.clamp(eigvals_r, min=floor)

    sqrt_diag = torch.sqrt(eigvals_clamped).to(work_dtype)
    inv_sqrt_diag = (1.0 / torch.sqrt(eigvals_clamped)).to(work_dtype)

    sqrt_G_M = (eigvecs * sqrt_diag.unsqueeze(0)) @ eigvecs.conj().T
    inv_sqrt_G_M = (eigvecs * inv_sqrt_diag.unsqueeze(0)) @ eigvecs.conj().T

    U_model = M @ inv_sqrt_G_M
    R_model = sqrt_G_M

    I = torch.eye(U_model.shape[1], device=U_model.device, dtype=U_model.dtype)
    ortho_residual = torch.linalg.matrix_norm(U_model.conj().T @ U_model - I, ord=2)
    factor_residual = torch.linalg.matrix_norm(U_model.conj().T @ M - R_model, ord=2)

    return {
        "M": M,
        "G_M": G_M,
        "U_model": U_model,
        "R_model": R_model,
        "inv_sqrt_G_M": inv_sqrt_G_M,
        "eigvals_clamped": eigvals_clamped,
        "ortho_residual": ortho_residual,
        "factor_residual": factor_residual,
    }


def compute_Sk_lambda(
    K: int,
    p: int,
    lam: complex,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build S_K^{(p)}(lambda) block-bidiagonal matrix.

    Shape: (Kp, (K+1)p), with -lam*I_p on block diagonal and I_p on first superdiagonal.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1. Got K={K}.")
    if p < 1:
        raise ValueError(f"p must be >= 1. Got p={p}.")

    if torch.is_complex(torch.empty((), dtype=dtype)):
        work_dtype = dtype
    else:
        work_dtype = complex_dtype_from_real(dtype)
    lam_c = torch.tensor(lam, device=device, dtype=work_dtype)
    I_p = torch.eye(p, device=device, dtype=work_dtype)

    S = torch.zeros(K * p, (K + 1) * p, device=device, dtype=work_dtype)
    for i in range(K):
        r0, r1 = i * p, (i + 1) * p
        c0, c1 = i * p, (i + 1) * p
        c2 = (i + 2) * p
        S[r0:r1, c0:c1] = -lam_c * I_p
        S[r0:r1, c1:c2] = I_p
    return S


def compute_N_LK_lambda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
) -> torch.Tensor:
    """Compute N_{L,K}(lambda) from (A,B,C,D) and manuscript definition."""
    if L < K + 1:
        raise ValueError(f"Need L >= K+1. Got L={L}, K={K}.")
    if K < 1:
        raise ValueError(f"K must be >= 1. Got K={K}.")

    m = B.shape[1]
    p = C.shape[0]
    n = A.shape[0]
    device = A.device
    dtype = A.dtype

    if torch.is_complex(torch.empty((), dtype=dtype)):
        work_dtype = dtype
    else:
        work_dtype = complex_dtype_from_real(dtype)

    A_c = to_complex(A, work_dtype)
    B_c = to_complex(B, work_dtype)
    C_c = to_complex(C, work_dtype)
    D_c = to_complex(D, work_dtype)

    T_K = compute_toeplitz_matrix(C=C_c, A=A_c, B=B_c, D=D_c, K=K + 1)  # ((K+1)p, (K+1)m)
    O_K = compute_observability_matrix(C=C_c, A=A_c, K=K + 1)  # ((K+1)p, n)

    S_sel = torch.zeros((K + 1) * m, L * m, device=device, dtype=work_dtype)
    S_sel[:, : (K + 1) * m] = torch.eye((K + 1) * m, device=device, dtype=work_dtype)

    block_left = T_K @ S_sel  # ((K+1)p, Lm)
    block = torch.cat([block_left, O_K], dim=1)  # ((K+1)p, Lm+n)

    S_K_lam = compute_Sk_lambda(K=K, p=p, lam=lam, dtype=work_dtype, device=device)  # (Kp, (K+1)p)
    N = S_K_lam @ block  # (Kp, Lm+n)
    return N


def compute_Q_LK_model(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute model-only Q_{L,K}(lambda) = N_{L,K}(lambda) R_model^{-1}."""
    model = compute_model_basis_R(C=C, A=A, B=B, D=D, L=L, K=K, eps=eps)
    R_model = model["R_model"]
    N = compute_N_LK_lambda(A=A, B=B, C=C, D=D, L=L, K=K, lam=lam)

    R_used = to_complex(R_model, N.dtype)
    Q = torch.linalg.solve(R_used.T, N.T).T
    return Q


def compute_basis_alignment(
    U_model: torch.Tensor,
    U_data: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute unitary Procrustes alignment W between two orthonormal bases."""
    if U_model.ndim != 2 or U_data.ndim != 2:
        raise ValueError(
            f"U_model and U_data must be matrices. Got {tuple(U_model.shape)} and {tuple(U_data.shape)}."
        )
    if U_model.shape != U_data.shape:
        raise ValueError(
            f"U_model and U_data must have same shape. Got {tuple(U_model.shape)} and {tuple(U_data.shape)}."
        )

    dtype = U_model.dtype if torch.is_complex(U_model) else complex_dtype_from_real(U_model.dtype)
    U_mod = to_complex(U_model, dtype)
    U_dat = to_complex(U_data, dtype)

    C = U_mod.conj().T @ U_dat
    P, singular_values, Vh = torch.linalg.svd(C, full_matrices=False)
    W_align = P @ Vh

    residual = torch.linalg.matrix_norm(U_dat - U_mod @ W_align, ord=2)
    return {
        "W_align": W_align,
        "singular_values": singular_values,
        "alignment_residual": residual,
    }


def aligned_q_error(
    Q_data: torch.Tensor,
    Q_model: torch.Tensor,
    W_align: torch.Tensor,
    ord: Union[int, str] = 2,
) -> torch.Tensor:
    """Compute ||Q_data - Q_model W_align|| under a fixed basis alignment."""
    if Q_data.ndim != 2 or Q_model.ndim != 2 or W_align.ndim != 2:
        raise ValueError("Q_data, Q_model, and W_align must be matrices.")
    if Q_data.shape[0] != Q_model.shape[0]:
        raise ValueError(
            f"Incompatible row dimensions: Q_data={tuple(Q_data.shape)}, Q_model={tuple(Q_model.shape)}."
        )
    if Q_model.shape[1] != W_align.shape[0] or Q_data.shape[1] != W_align.shape[1]:
        raise ValueError(
            f"Incompatible alignment shape: Q_data={tuple(Q_data.shape)}, "
            f"Q_model={tuple(Q_model.shape)}, W_align={tuple(W_align.shape)}."
        )

    dtype = Q_data.dtype if torch.is_complex(Q_data) else complex_dtype_from_real(Q_data.dtype)
    Qd = to_complex(Q_data, dtype)
    Qm = to_complex(Q_model, dtype)
    W = to_complex(W_align, dtype)
    return torch.linalg.matrix_norm(Qd - Qm @ W, ord=ord)


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
        υ  = Υ U_y                          (reduced signal)
        K^y = (∫ υ υ^* dt)^{-1} ∫ υ dυ^*

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

    # 3. Reduced signal: υ = Υ U_y (row-stacked samples)
    upsilon = Upsilon @ U_y  # (N', r)
    upsilon = to_complex(upsilon, complex_dtype)

    # 4. Build K^y = (∫ υ υ^* dt)^{-1} · ∫ υ dυ^*
    d_upsilon = upsilon[1:] - upsilon[:-1]  # Δυ (no /dt; cancels)
    upsilon0 = upsilon[:-1]

    G = upsilon0.T @ upsilon0.conj() * dt   # ∫ υ υ^* dt
    G = 0.5 * (G + G.conj().T)
    M = upsilon0.T @ d_upsilon.conj()       # ∫ υ dυ^* (dt cancels)

    # Use lstsq for robustness when G is ill-conditioned
    K_matrix = torch.linalg.lstsq(G, M).solution  # G^{-1} M
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


def check_controllability_observable_quotient(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    dt: float,
    candidate_lambdas: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    rank_tol: float = 1e-8,
) -> dict:
    """Check controllability via row-rank of Q_{L,K}(λ).

    Uses thresholded rank based on singular values:
        rank(Q_{L,K}(λ)) = number of σ_i(Q) > threshold.
    """
    p = y.shape[1]
    expected_rank = K * p

    coords = compute_observable_quotient_coordinates(
        u=u,
        y=y,
        L=L,
        K=K,
        n=n,
        dt=dt,
        rank_tol=rank_tol,
    )
    xi = coords["xi"]
    gamma_xi = coords["Gamma_xi"]

    if candidate_lambdas is None:
        _, candidate_lambdas = compute_K_LK_reduced(
            y, L, K, dt, rank_tol=rank_tol
        )

    ranks = []
    min_svals = []
    h_factorization_residuals = []

    for lam in candidate_lambdas:
        lam_val = lam.item() if torch.is_tensor(lam) else lam
        Q = compute_Q_LK_from_coordinates(
            y=y,
            K=K,
            lam=lam_val,
            dt=dt,
            xi=xi,
            Gamma_xi=gamma_xi,
        )
        svals = torch.linalg.svdvals(Q).real
        rank = int((svals > threshold).sum().item())
        ranks.append(rank)
        if svals.numel() >= expected_rank:
            min_svals.append(float(svals[expected_rank - 1].item()))
        else:
            min_svals.append(0.0)

        # Diagnostic only: consistency of H = Q Gamma_xi Q^* on aligned samples.
        lambda_K_y_lam = compute_filtered_derivative_lift(y, K, lam_val, dt)
        lambda_K_y_lam = to_complex(lambda_K_y_lam, Q.dtype)
        n_common = min(lambda_K_y_lam.shape[0], xi.shape[0])
        y_lam = lambda_K_y_lam[:n_common]
        xi_used = xi[:n_common]
        gamma_used = xi_used.T @ xi_used.conj() * dt
        gamma_used = 0.5 * (gamma_used + gamma_used.conj().T)
        H_direct = y_lam.T @ y_lam.conj() * dt
        H_fact = Q @ gamma_used @ Q.conj().T
        residual = torch.linalg.matrix_norm(H_direct - H_fact, ord=2).item()
        h_factorization_residuals.append(float(residual))

    ranks_t = torch.tensor(ranks)
    min_svals_t = torch.tensor(min_svals)
    residuals_t = torch.tensor(h_factorization_residuals)
    is_controllable = bool((ranks_t >= expected_rank).all().item())

    return {
        "is_controllable": is_controllable,
        "expected_rank": expected_rank,
        "ranks": ranks_t,
        "min_singular_values": min_svals_t,
        "candidate_lambdas": candidate_lambdas,
        "target_coordinate_rank": coords["target_rank"],
        "numerical_coordinate_rank": coords["numerical_rank"],
        "coordinate_condition_number": coords["condition_number"],
        "h_factorization_residuals": residuals_t,
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

"""Hautus test computations for controllability analysis.

This module implements the data-driven Hautus tests for continuous-time systems,
including time-domain and frequency-domain formulations.

References:
    - Theorem 1: Continuous Hautus Test (time-domain Gramian)
    - Corollary 1: Continuous Hautus test via cross-moments
    - Proposition 1: Cross-moment error bound
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional

from .utils import complex_dtype_from_real, to_complex, stack_z, ensure_dt_vector
from .gramians import gramian_Sz_time, integral_xxH_time, integral_xdot_xH_time


def cross_moment_H_time(
    x: torch.Tensor,
    u: torch.Tensor,
    lam: Union[complex, torch.Tensor],
    dt: Union[float, torch.Tensor],
    return_parts: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""Compute cross-moment H_λ(T) in time domain (Eq. 10 in manuscript).
    
    H_λ(T) = ∫ dy_λ(t) z(t)^T
    
    where dy_λ(t) = dx(t) - λ x(t) dt and z = [x; u].
    
    Discrete approximation (left-point):
        H_λ ≈ Σ_{k=0}^{N-2} (Δx_k - λ x_k Δt_k) z_k^T

    Args:
        x: State trajectory of shape (N, n)
        u: Input trajectory of shape (N, m)
        lam: Complex scalar λ
        dt: Scalar float or tensor of shape (N-1,)
        return_parts: If True, returns (H_λ, H_dx, H_xdt) separately

    Returns:
        H_λ: Cross-moment matrix of shape (n, n+m)
        
    If return_parts=True:
        H_λ: Cross-moment matrix
        H_dx: Σ Δx_k z_k^T
        H_xdt: Σ (x_k Δt_k) z_k^T
    """
    assert x.ndim == 2 and u.ndim == 2, "Expected x:(N,n), u:(N,m)"
    assert x.shape[0] == u.shape[0], "x and u must have the same length N"

    device = x.device
    real_dtype = x.dtype
    complex_dtype = complex_dtype_from_real(real_dtype)

    N, n = x.shape
    m = u.shape[1]
    assert N >= 2, "Need at least 2 samples"

    lam_c = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=device)
    lam_c = to_complex(lam_c, complex_dtype)

    dx = x[1:] - x[:-1]          # (N-1, n)
    x0 = x[:-1]                  # (N-1, n)
    u0 = u[:-1]                  # (N-1, m)
    z0 = stack_z(x0, u0)         # (N-1, n+m)

    dt_vec = ensure_dt_vector(dt, N, device, real_dtype)

    dx_c = to_complex(dx, complex_dtype)
    x0_c = to_complex(x0, complex_dtype)
    z0_c = to_complex(z0, complex_dtype)

    # H_dx = Σ Δx_k z_k^T
    H_dx = torch.einsum("kn,kp->np", dx_c, z0_c)

    # H_xdt = Σ (x_k Δt_k) z_k^T
    xdt = x0_c * dt_vec[:, None].to(complex_dtype)
    H_xdt = torch.einsum("kn,kp->np", xdt, z0_c)

    H_lam = H_dx - lam_c * H_xdt

    if return_parts:
        return H_lam, H_dx, H_xdt
    return H_lam


def cross_moment_H_fft(
    x: torch.Tensor,
    u: torch.Tensor,
    lam: Union[complex, torch.Tensor],
    dt: float,
    omega_max: Optional[float] = None,
    return_parts: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""Compute cross-moment H_λ using FFT (frequency domain).
    
    Uses Parseval's theorem to compute H_λ from Fourier transforms,
    avoiding explicit derivative computation.
    
    ŷ_λ(iω) = x(T)e^{-iωT} - x(0) + (iω - λ)x̂(iω)

    Args:
        x: State trajectory of shape (N, n)
        u: Input trajectory of shape (N, m)
        lam: Complex scalar λ
        dt: Time step (scalar)
        omega_max: Optional frequency cutoff
        return_parts: If True, returns additional components

    Returns:
        H_λ: Cross-moment matrix of shape (n, n+m)
        S_Z: Gramian from FFT of shape (n+m, n+m)
        
    If return_parts=True, also returns:
        H_0: ∫ dx z^*
        H_x: ∫ x z^*
    """
    assert x.ndim == 2 and u.ndim == 2
    assert x.shape[0] == u.shape[0]
    
    device = x.device
    real_dtype = x.dtype
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    N, n = x.shape
    m = u.shape[1]
    p = n + m

    lam_c = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=device)
    lam_c = lam_c.to(dtype=complex_dtype)

    # z(t) = [x; u]
    z = torch.cat([x, u], dim=1)  # (N, p)

    # Frequency grid (angular)
    omega = 2.0 * np.pi * torch.fft.fftfreq(N, d=dt).to(device=device, dtype=real_dtype)

    # FFT approximations
    Xhat = dt * torch.fft.fft(x.to(complex_dtype), dim=0)  # (N, n)
    Zhat = dt * torch.fft.fft(z.to(complex_dtype), dim=0)  # (N, p)

    # Approximate ∫ e^{-iωt} dx(t) by FFT of increments
    dx = torch.zeros_like(x)
    dx[:-1] = x[1:] - x[:-1]
    DXhat = torch.fft.fft(dx.to(complex_dtype), dim=0)  # (N, n), no dt factor

    # dŷ_λ = (∫ e^{-iωt} dx) - λ (∫ e^{-iωt} x dt)
    DYhat = DXhat - lam_c * Xhat  # (N, n)

    # Optional frequency cutoff
    if omega_max is not None:
        mask = omega.abs() <= omega_max
        DYhat = DYhat[mask]
        Xhat = Xhat[mask]
        Zhat = Zhat[mask]
        DXhat = DXhat[mask]

    # Parseval scaling: Δω/(2π) = 1/(N dt)
    scale = 1.0 / (N * dt)

    # H_λ = ∫ dy_λ z^* ≈ Σ DYhat Zhat^* Δω/(2π)
    H_lam = (DYhat.T @ Zhat.conj()) * scale  # (n, p)

    # S_Z from FFT
    Sz = (Zhat.T @ Zhat.conj()) * scale  # (p, p)

    if return_parts:
        H0 = (DXhat.T @ Zhat.conj()) * scale
        Hx = (Xhat.T @ Zhat.conj()) * scale
        return H_lam, H0, Hx, Sz

    return H_lam, Sz


def gramian_G_from_H(
    H: torch.Tensor,
    Sz: torch.Tensor,
    ridge: float = 0.0,
) -> torch.Tensor:
    r"""Compute Gramian G_λ from cross-moment H (Eq. 15).
    
    G_λ = H S_Z^{-1} H^*
    
    Computed stably via solve rather than explicit inverse.

    Args:
        H: Cross-moment matrix of shape (n, n+m)
        Sz: Stacked Gramian of shape (n+m, n+m)
        ridge: Regularization parameter

    Returns:
        G_λ: Gramian of shape (n, n)
    """
    p = Sz.shape[-1]
    complex_dtype = complex_dtype_from_real(Sz.dtype)
    Sz_reg = Sz.to(complex_dtype) + ridge * torch.eye(p, device=Sz.device, dtype=complex_dtype)

    # Solve Sz_reg * X = H^*
    X = torch.linalg.solve(Sz_reg, H.conj().T)  # (p, n)
    return H @ X  # (n, n)


def estimate_Hautus_matrix(
    H: torch.Tensor,
    Sz: torch.Tensor,
    ridge: float = 0.0,
) -> torch.Tensor:
    r"""Estimate the Hautus matrix P_λ from data (Eq. 17).
    
    P̂_λ(T) = H_λ(T) S_Z(u)^{-1}
    
    This provides a data-driven estimate of [A - λI, B].

    Args:
        H: Cross-moment matrix of shape (n, n+m)
        Sz: Stacked Gramian of shape (n+m, n+m)
        ridge: Regularization parameter

    Returns:
        P̂_λ: Estimated Hautus matrix of shape (n, n+m)
    """
    p = Sz.shape[-1]
    complex_dtype = complex_dtype_from_real(Sz.dtype)
    H = H.to(dtype=complex_dtype)
    Sz_reg = Sz.to(complex_dtype) + torch.eye(p, device=Sz.device, dtype=complex_dtype) * ridge
    
    return torch.linalg.solve(Sz_reg, H.conj().T).conj().T


def true_Hautus_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
    lam: Union[complex, torch.Tensor],
) -> torch.Tensor:
    r"""Compute the true Hautus matrix P_λ = [A - λI, B].
    
    Args:
        A: System matrix of shape (n, n)
        B: Input matrix of shape (n, m)
        lam: Complex scalar λ

    Returns:
        P_λ: Hautus matrix of shape (n, n+m)
    """
    n = A.shape[0]
    device = A.device
    complex_dtype = complex_dtype_from_real(A.dtype)
    
    lam_c = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=device)
    lam_c = to_complex(lam_c, complex_dtype)
    
    A_c = to_complex(A, complex_dtype)
    B_c = to_complex(B, complex_dtype)
    I = torch.eye(n, device=device, dtype=complex_dtype)
    
    return torch.cat([A_c - lam_c * I, B_c], dim=1)


def hautus_test(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    lam: Union[complex, torch.Tensor],
    ridge: float = 1e-8,
    method: str = "time",
    omega_max: Optional[float] = None,
) -> dict:
    r"""Perform data-driven Hautus test for a given λ.
    
    Computes the estimated Hautus matrix and its minimum singular value,
    which characterizes controllability margin.

    Args:
        x: State trajectory of shape (N, n)
        u: Input trajectory of shape (N, m)
        dt: Time step
        lam: Complex scalar λ to test
        ridge: Regularization parameter
        method: "time" or "fft"
        omega_max: Frequency cutoff for FFT method

    Returns:
        Dictionary containing:
            - P_hat: Estimated Hautus matrix
            - sigma_min: Minimum singular value
            - H: Cross-moment matrix
            - Sz: Stacked Gramian
    """
    if method == "time":
        H = cross_moment_H_time(x, u, lam=lam, dt=dt)
        Sz = gramian_Sz_time(x, u, dt=dt)
    elif method == "fft":
        H, Sz = cross_moment_H_fft(x, u, lam=lam, dt=dt, omega_max=omega_max)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    P_hat = estimate_Hautus_matrix(H, Sz, ridge=ridge)
    sigma_min = torch.linalg.svdvals(P_hat).min().item()
    
    return {
        "P_hat": P_hat,
        "sigma_min": sigma_min,
        "H": H,
        "Sz": Sz,
    }


def check_controllability(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    ridge: float = 1e-8,
    method: str = "time",
    omega_max: Optional[float] = None,
    return_details: bool = False,
) -> Union[bool, dict]:
    r"""Check controllability using finite candidate set (Corollary 2).
    
    Computes candidate eigenvalues from K = S_X^{-1} M_X and checks
    the Hautus condition only at these candidates.

    Args:
        x: State trajectory of shape (N, n)
        u: Input trajectory of shape (N, m)
        dt: Time step
        ridge: Regularization parameter
        method: "time" or "fft"
        omega_max: Frequency cutoff for FFT method
        return_details: If True, return detailed results

    Returns:
        is_controllable: Boolean indicating controllability
        
    If return_details=True:
        Dictionary containing:
            - is_controllable: Boolean
            - candidate_lambdas: Candidate eigenvalues
            - sigma_mins: Minimum singular values at each candidate
            - min_sigma: Overall minimum singular value
    """
    from .gramians import compute_candidate_eigenvalues
    
    # Get candidate eigenvalues
    lambdas = compute_candidate_eigenvalues(x, dt)
    
    sigma_mins = []
    for lam in lambdas:
        result = hautus_test(x, u, dt, lam, ridge=ridge, method=method, omega_max=omega_max)
        sigma_mins.append(result["sigma_min"])
    
    sigma_mins = torch.tensor(sigma_mins)
    min_sigma = sigma_mins.min().item()
    is_controllable = min_sigma > ridge  # Use ridge as threshold
    
    if return_details:
        return {
            "is_controllable": is_controllable,
            "candidate_lambdas": lambdas,
            "sigma_mins": sigma_mins,
            "min_sigma": min_sigma,
        }
    
    return is_controllable


def compare_with_true(
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    lam: Union[complex, torch.Tensor],
    ridge: float = 1e-8,
    method: str = "time",
    omega_max: Optional[float] = None,
) -> dict:
    r"""Compare estimated Hautus matrix with true value.
    
    Computes both the estimated P̂_λ and true P_λ = [A - λI, B],
    along with error metrics.

    Args:
        A: True system matrix (n, n)
        B: True input matrix (n, m)
        x: State trajectory (N, n)
        u: Input trajectory (N, m)
        dt: Time step
        lam: Complex scalar λ
        ridge: Regularization parameter
        method: "time" or "fft"
        omega_max: Frequency cutoff for FFT method

    Returns:
        Dictionary containing:
            - P_true: True Hautus matrix
            - P_hat: Estimated Hautus matrix
            - error_norm: ||P̂ - P||_2
            - sigma_min_true: σ_min(P)
            - sigma_min_hat: σ_min(P̂)
            - sigma_min_error: |σ_min(P̂) - σ_min(P)|
    """
    result = hautus_test(x, u, dt, lam, ridge=ridge, method=method, omega_max=omega_max)
    P_hat = result["P_hat"]
    
    P_true = true_Hautus_matrix(A, B, lam)
    
    # Ensure same dtype for comparison
    complex_dtype = P_hat.dtype
    P_true = P_true.to(complex_dtype)
    
    error_norm = torch.linalg.norm(P_hat - P_true, ord=2).item()
    sigma_min_true = torch.linalg.svdvals(P_true).min().item()
    sigma_min_hat = torch.linalg.svdvals(P_hat).min().item()
    
    return {
        "P_true": P_true,
        "P_hat": P_hat,
        "error_norm": error_norm,
        "sigma_min_true": sigma_min_true,
        "sigma_min_hat": sigma_min_hat,
        "sigma_min_error": abs(sigma_min_hat - sigma_min_true),
    }

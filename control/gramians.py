"""Gramian computation functions for controllability analysis.

This module implements time-domain and frequency-domain Gramian computations
as described in the data-driven Hautus tests for continuous-time systems.
"""

import torch
from typing import Union, Tuple

from .utils import complex_dtype_from_real, to_complex, stack_z, ensure_dt_vector


def gramian_Sz_time(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Compute the stacked signal Gramian S_Z in time domain.
    
    S_Z = ∫ z z^T dt  ≈  Σ_{k=0}^{N-2} z_k z_k^T Δt_k
    
    where z_k = [x_k; u_k] (left-point rule).

    Args:
        x: State trajectory of shape (N, n)
        u: Input trajectory of shape (N, m)
        dt: Scalar float or tensor of shape (N-1,) of time steps

    Returns:
        S_Z: Gramian matrix of shape (n+m, n+m)
    """
    assert x.ndim == 2 and u.ndim == 2, "Expected x:(N,n), u:(N,m)"
    assert x.shape[0] == u.shape[0], "x and u must have the same length N"

    N = x.shape[0]
    assert N >= 2, "Need at least 2 samples"

    z = stack_z(x[:-1], u[:-1])  # (N-1, p)
    dt_vec = ensure_dt_vector(dt, N, z.device, z.dtype)

    # Sz = Σ dt_k z_k z_k^T
    Sz = z.T @ (z * dt_vec[:, None])  # (p, p)
    return Sz


def integral_xxH_time(
    x: torch.Tensor,
    dt: Union[float, int, torch.Tensor],
) -> torch.Tensor:
    r"""Compute ∫_0^T x(t) x(t)^* dt using left-point rule.
    
    This is the state-only Gramian S_X.

    Args:
        x: State trajectory of shape (N, n), real or complex
        dt: Scalar float/int or tensor of shape (N-1,)

    Returns:
        S_X: Gramian matrix of shape (n, n)
    """
    assert x.ndim == 2, "Expected x:(N,n)"
    N, n = x.shape
    assert N >= 2, "Need at least 2 samples"

    x0 = x[:-1]  # (N-1, n) left-point samples
    dt_vec = ensure_dt_vector(dt, N, x.device, torch.float32)

    # Sx = Σ dt_k x_k x_k^*
    Sx = x0.conj().T @ (x0 * dt_vec[:, None].to(x0.dtype))
    return Sx


def integral_xdot_xH_time(
    x: torch.Tensor,
    dt: Union[float, int, torch.Tensor],
    use_dx: bool = True,
) -> torch.Tensor:
    r"""Compute ∫_0^T ẋ(t) x(t)^* dt using left-point rule.
    
    Two equivalent discretizations:
      - use_dx=True (recommended): ẋ_k Δt_k ≈ Δx_k => Σ Δx_k x_k^*
      - use_dx=False: ẋ_k ≈ Δx_k/Δt_k => Σ (Δx_k/Δt_k) x_k^* Δt_k

    Args:
        x: State trajectory of shape (N, n), real or complex
        dt: Scalar float/int or tensor of shape (N-1,)
        use_dx: If True, avoids division by dt (more stable)

    Returns:
        M: Cross-moment matrix of shape (n, n)
    """
    assert x.ndim == 2, "Expected x:(N,n)"
    N, n = x.shape
    assert N >= 2, "Need at least 2 samples"

    x0 = x[:-1]          # (N-1, n) left point
    dx = x[1:] - x[:-1]  # (N-1, n)
    dt_vec = ensure_dt_vector(dt, N, x.device, torch.float32)

    if use_dx:
        # M = Σ Δx_k x_k^*
        M = dx.T @ x0.conj()
    else:
        # ẋ_k ≈ Δx_k / Δt_k
        xdot = dx / dt_vec[:, None].to(dx.dtype)
        # M = Σ ẋ_k x_k^* Δt_k
        M = xdot.T @ (x0.conj() * dt_vec[:, None].to(x0.dtype))

    return M


def compute_candidate_eigenvalues(
    x: torch.Tensor,
    dt: Union[float, int, torch.Tensor],
) -> torch.Tensor:
    r"""Compute the candidate eigenvalues λ from Theorem 4 (finite candidate set).
    
    The matrix K = (∫ x x^* dt)^{-1} (∫ ẋ x^* dt) has the property that
    rank failure of the Hautus pencil can only occur at λ ∈ σ(K).

    Args:
        x: State trajectory of shape (N, n)
        dt: Scalar float/int or tensor of shape (N-1,)

    Returns:
        eigenvalues: Complex tensor of shape (n,) containing σ(K)
    """
    Sx = integral_xxH_time(x, dt)
    Mx = integral_xdot_xH_time(x, dt)
    
    K = torch.linalg.solve(Sx, Mx.T).T  # More stable than inv(Sx) @ Mx
    return torch.linalg.eigvals(K)


def gramian_Sx_from_fft(
    x: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    r"""Compute state Gramian S_X from FFT (Parseval).
    
    S_X = ∫ x(t) x(t)^* dt = ∫ X̂(iω) X̂(iω)^* dω/(2π)

    Args:
        x: State trajectory of shape (N, n)
        dt: Time step

    Returns:
        S_X: Gramian matrix of shape (n, n)
    """
    N, n = x.shape
    complex_dtype = complex_dtype_from_real(x.dtype)
    
    # FFT with dt scaling
    Xhat = dt * torch.fft.fft(x.to(complex_dtype), dim=0)  # (N, n)
    
    # Parseval: Δω/(2π) = 1/(N dt)
    scale = 1.0 / (N * dt)
    Sx = (Xhat.conj().T @ Xhat) * scale
    return Sx

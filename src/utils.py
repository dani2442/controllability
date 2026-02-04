"""Utility functions for controllability analysis."""

import torch
from typing import Union


def complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    """Convert a real dtype to its complex counterpart."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def to_complex(x: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor to complex dtype if not already complex."""
    return x if torch.is_complex(x) else x.to(complex_dtype)


def stack_z(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Stack state and input into augmented state z = [x; u].
    
    Args:
        x: State tensor of shape (N, n) or (batch, n)
        u: Input tensor of shape (N, m) or (batch, m)
        
    Returns:
        Stacked tensor of shape (N, n+m) or (batch, n+m)
    """
    return torch.cat([x, u], dim=-1)


def ensure_dt_vector(
    dt: Union[float, int, torch.Tensor],
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Ensure dt is a vector of length N-1.
    
    Args:
        dt: Scalar float/int or tensor of shape (N-1,)
        N: Number of time samples
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tensor of shape (N-1,)
    """
    if isinstance(dt, (float, int)):
        return torch.full((N - 1,), float(dt), device=device, dtype=dtype)
    else:
        dt_vec = dt.to(device=device, dtype=dtype)
        assert dt_vec.shape == (N - 1,), f"dt must have shape (N-1,) = {(N-1,)}, got {dt_vec.shape}"
        return dt_vec


def make_stable_A(n: int, device: torch.device, margin: float = 0.1) -> torch.Tensor:
    """Create a random stable matrix A.
    
    Args:
        n: State dimension
        device: Target device
        margin: Stability margin (shift eigenvalues by this amount)
        
    Returns:
        Stable matrix A of shape (n, n)
    """
    A = torch.randn(n, n, device=device)
    # Shift eigenvalues to ensure stability
    A = A - (torch.linalg.eigvals(A).real.max() + margin) * torch.eye(n, device=device, dtype=A.dtype)
    return A

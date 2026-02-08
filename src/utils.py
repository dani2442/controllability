"""Utility functions for controllability analysis."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    """Convert a real dtype to its complex counterpart."""
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def to_complex(x: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor to complex dtype if not already complex."""
    return x if torch.is_complex(x) else x.to(complex_dtype)


def generate_stable_system(
    n: int,
    m: int,
    p: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    stability_margin: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random stable LTI system (A, B, C, D).
    
    Creates matrices such that all eigenvalues of A have negative real parts.
    
    System:
        dx = (Ax + Bu)dt + process_noise
        y  = Cx + Du + measurement_noise
    
    Args:
        n: State dimension
        m: Input dimension
        p: Output dimension
        device: Target device
        dtype: Data type
        stability_margin: Minimum distance of eigenvalues from imaginary axis
        
    Returns:
        A: System matrix (n, n) with all eigenvalues having Re(λ) < -stability_margin
        B: Input matrix (n, m)
        C: Output matrix (p, n)
        D: Feedthrough matrix (p, m)
    """
    if device is None:
        device = torch.device("cpu")
    
    # Generate random A and shift eigenvalues to ensure stability
    A = torch.randn(n, n, device=device, dtype=dtype)
    eigvals = torch.linalg.eigvals(A)
    max_real = eigvals.real.max().item()
    # Shift so that max real part is at -stability_margin
    A = A - (max_real + stability_margin) * torch.eye(n, device=device, dtype=dtype)
    
    # Random input, output, and feedthrough matrices
    B = torch.randn(n, m, device=device, dtype=dtype)
    C = torch.randn(p, n, device=device, dtype=dtype)
    D = torch.randn(p, m, device=device, dtype=dtype) * 0.1  # Small feedthrough
    
    return A, B, C, D


def compute_toeplitz_matrix(
    C: torch.Tensor,
    A: torch.Tensor, 
    B: torch.Tensor,
    D: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Compute the Toeplitz matrix T_K for the system.
    
    T_K has structure:
        [D,          0,          ..., 0      ]
        [CB,         D,          ..., 0      ]
        [CAB,        CB,         ..., 0      ]
        [...]
        [CA^{K-2}B,  CA^{K-3}B,  ..., D      ]
    
    Args:
        C: Output matrix (p, n)
        A: System matrix (n, n)
        B: Input matrix (n, m)
        D: Feedthrough matrix (p, m)
        K: Number of output derivative blocks
        
    Returns:
        T_K: Toeplitz matrix of shape (Kp, Km)
    """
    p, n = C.shape
    m = B.shape[1]
    
    device = C.device
    dtype = C.dtype
    
    T = torch.zeros(K * p, K * m, device=device, dtype=dtype)
    
    # Fill the Toeplitz structure
    # Markov parameters: D, CB, CAB, CA^2B, ...
    markov = [D]  # (p, m)
    Ak = torch.eye(n, device=device, dtype=dtype)
    for k in range(K - 1):
        markov.append(C @ Ak @ B)  # CA^k B
        Ak = Ak @ A
    
    # Place Markov parameters in Toeplitz structure
    for row in range(K):
        for col in range(row + 1):
            T[row*p:(row+1)*p, col*m:(col+1)*m] = markov[row - col]
    
    return T


def compute_observability_matrix(C: torch.Tensor, A: torch.Tensor, K: int) -> torch.Tensor:
    """Compute the observability matrix O_K.
    
    O_K = [C; CA; CA^2; ...; CA^{K-1}]
    
    Args:
        C: Output matrix (p, n)
        A: System matrix (n, n)
        K: Number of blocks
        
    Returns:
        O_K: Observability matrix of shape (Kp, n)
    """
    p, n = C.shape
    device = C.device
    dtype = C.dtype
    
    O = torch.zeros(K * p, n, device=device, dtype=dtype)
    Ak = torch.eye(n, device=device, dtype=dtype)
    
    for k in range(K):
        O[k*p:(k+1)*p, :] = C @ Ak
        Ak = Ak @ A
    
    return O


def compute_observability_index(C: torch.Tensor, A: torch.Tensor) -> int:
    """Compute the observability index ell(B).

    Uses the manuscript convention:

        O_k = [C; CA; ...; CA^k],
        ell(B) = min{k >= 1 : rank(O_k) = rank(O_{k-1})}.

    Args:
        C: Output matrix (p, n)
        A: System matrix (n, n)

    Returns:
        ell: The smallest k where the observability rank saturates
    """
    n = A.shape[0]
    device = A.device
    dtype = A.dtype

    # Start from O_0 = C.
    blocks = [C]
    rank_prev = torch.linalg.matrix_rank(C).item()
    Ak = A.clone()

    # Grow O_k by appending C A^k and detect the first rank saturation.
    for k in range(1, n + 1):
        blocks.append(C @ Ak)
        O_k = torch.cat(blocks, dim=0)
        rank_k = torch.linalg.matrix_rank(O_k).item()
        if rank_k == rank_prev:
            return k
        rank_prev = rank_k
        Ak = Ak @ A

    return n


def compute_lift_matrix(
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
) -> torch.Tensor:
    """Compute the lift matrix M_{L,K}.
    
    M_{L,K} = [I_{Lm},           0       ]
              [T_{K-1} S_{K,L},  O_{K-1} ]
    
    where S_{K,L} = [I_{Km}, 0] is the selector projecting onto first K blocks.
    
    Args:
        C: Output matrix (p, n)
        A: System matrix (n, n)
        B: Input matrix (n, m)
        D: Feedthrough matrix (p, m)
        L: Number of input derivative blocks
        K: Number of output derivative blocks
        
    Returns:
        M_{L,K}: Lift matrix of shape (Lm + Kp, Lm + n)
    """
    p, n = C.shape
    m = B.shape[1]
    device = C.device
    dtype = C.dtype
    
    # Dimensions
    rows = L * m + K * p
    cols = L * m + n
    
    M = torch.zeros(rows, cols, device=device, dtype=dtype)
    
    # Top-left: I_{Lm}
    M[:L*m, :L*m] = torch.eye(L * m, device=device, dtype=dtype)
    
    # Bottom-left: T_{K-1} @ S_{K,L} (Toeplitz times selector)
    # S_{K,L} selects first K*m columns from L*m columns
    T_K = compute_toeplitz_matrix(C, A, B, D, K)  # (Kp, Km)
    M[L*m:, :K*m] = T_K
    
    # Bottom-right: O_{K-1} (observability matrix)
    O_K = compute_observability_matrix(C, A, K)  # (Kp, n)
    M[L*m:, L*m:] = O_K
    
    return M


def smooth_signal(
    signal: torch.Tensor,
    window_size: int = 9,
    sigma: Optional[float] = None,
    mode: str = "gaussian",
    padding: str = "reflect",
) -> torch.Tensor:
    """Smooth a 1D time-series signal using a windowed convolution.

    Args:
        signal: Input signal (N,) or (N, d).
        window_size: Size of the smoothing window (odd integer).
        sigma: Gaussian std dev. If None and mode="gaussian", uses window_size / 6.
        mode: "gaussian" or "moving_average".
        padding: Padding mode for convolution ("reflect", "replicate", "constant").

    Returns:
        Smoothed signal with the same shape as input.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for symmetric smoothing.")

    squeeze_output = False
    if signal.ndim == 1:
        signal_2d = signal[:, None]
        squeeze_output = True
    elif signal.ndim == 2:
        signal_2d = signal
    else:
        raise ValueError("signal must have shape (N,) or (N, d).")

    if window_size == 1:
        return signal.clone()

    n_samples = signal_2d.shape[0]
    pad = window_size // 2
    pad_mode = padding
    if pad_mode == "reflect" and n_samples <= pad:
        pad_mode = "replicate"

    device = signal.device
    dtype = signal.dtype

    if mode == "gaussian":
        if sigma is None:
            sigma = window_size / 6.0
        x = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    elif mode == "moving_average":
        kernel = torch.ones(window_size, device=device, dtype=dtype)
    else:
        raise ValueError("mode must be 'gaussian' or 'moving_average'.")

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    # Convolve each channel independently
    signal_t = signal_2d.T.unsqueeze(0)  # (1, d, N)
    kernel_t = kernel.repeat(signal_t.shape[1], 1, 1)  # (d, 1, window_size)

    signal_pad = F.pad(signal_t, (pad, pad), mode=pad_mode)
    smoothed = F.conv1d(signal_pad, kernel_t, groups=signal_t.shape[1])
    smoothed = smoothed.squeeze(0).T  # (N, d)

    if squeeze_output:
        return smoothed.squeeze(1)
    return smoothed

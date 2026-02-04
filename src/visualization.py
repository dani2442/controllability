"""Visualization functions for controllability analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, Optional, List


def plot_trajectories(
    ts: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """Plot state, input, and output trajectories.
    
    Args:
        ts: Time grid (N,)
        x: State trajectory (N, n)
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    ts_np = ts.cpu().numpy()
    x_np = x.cpu().numpy()
    u_np = u.cpu().numpy()
    y_np = y.cpu().numpy()
    
    n = x_np.shape[1]
    m = u_np.shape[1]
    p = y_np.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # States
    ax = axes[0]
    for i in range(min(n, 5)):
        ax.plot(ts_np, x_np[:, i], label=f"$x_{{{i+1}}}$", alpha=0.8)
    ax.set_ylabel("States $x$")
    ax.legend(loc="upper right", ncol=min(n, 5))
    ax.grid(True, alpha=0.3)
    ax.set_title("System Trajectories")
    
    # Inputs
    ax = axes[1]
    for i in range(m):
        ax.plot(ts_np, u_np[:, i], label=f"$u_{{{i+1}}}$", alpha=0.8)
    ax.set_ylabel("Inputs $u$")
    ax.legend(loc="upper right", ncol=m)
    ax.grid(True, alpha=0.3)
    
    # Outputs
    ax = axes[2]
    for i in range(min(p, 5)):
        ax.plot(ts_np, y_np[:, i], label=f"$y_{{{i+1}}}$", alpha=0.8)
    ax.set_ylabel("Outputs $y$")
    ax.set_xlabel("Time $t$")
    ax.legend(loc="upper right", ncol=min(p, 5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_eigenvalues(
    eigenvalues: torch.Tensor,
    title: str = "Eigenvalue Spectrum",
    figsize: Tuple[int, int] = (8, 6),
    marker_size: float = 100,
) -> Figure:
    """Plot eigenvalues in the complex plane.
    
    Args:
        eigenvalues: Complex eigenvalues
        title: Plot title
        figsize: Figure size
        marker_size: Size of markers
        
    Returns:
        fig: Matplotlib figure
    """
    if torch.is_tensor(eigenvalues):
        eigenvalues = eigenvalues.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=marker_size, 
               c='blue', alpha=0.7, edgecolors='black', linewidths=1)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_gramian_eigenvalues(
    eigenvalues_list: List[torch.Tensor],
    lambda_labels: List[str],
    expected_rank: int,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot eigenvalues of G_{L,K}(λ) for multiple λ values.
    
    Args:
        eigenvalues_list: List of eigenvalue tensors, one per λ
        lambda_labels: Labels for each λ
        expected_rank: Expected rank (Lm + n)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(eigenvalues_list)))
    
    for i, (eigvals, label) in enumerate(zip(eigenvalues_list, lambda_labels)):
        if torch.is_tensor(eigvals):
            eigvals = eigvals.cpu().numpy()
        eigvals_sorted = np.sort(eigvals)[::-1]
        
        indices = np.arange(1, len(eigvals_sorted) + 1)
        ax.semilogy(indices, eigvals_sorted + 1e-16, 'o-', 
                    color=colors[i], label=label, alpha=0.7, markersize=6)
    
    # Mark expected rank position
    ax.axvline(x=expected_rank, color='red', linestyle='--', 
               label=f'$Lm+n = {expected_rank}$', linewidth=2)
    
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel(r"$\lambda_i(\mathbf{G}_{L,K})$")
    ax.set_title(r"Eigenvalue Decay of $\mathbf{G}_{L,K}(\lambda)$")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_controllability_margin(
    lambdas: torch.Tensor,
    margins: torch.Tensor,
    threshold: float = 1e-6,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot controllability margin (λ_{Lm+n} eigenvalue) vs λ.
    
    Args:
        lambdas: Candidate λ values (complex)
        margins: Controllability margins at each λ
        threshold: Threshold for controllability
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    if torch.is_tensor(lambdas):
        lambdas = lambdas.cpu().numpy()
    if torch.is_tensor(margins):
        margins = margins.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: margins in complex plane
    ax = axes[0]
    scatter = ax.scatter(lambdas.real, lambdas.imag, c=margins, 
                         cmap='RdYlGn', s=100, edgecolors='black', linewidths=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title(r"Controllability Margin at Candidate $\lambda$")
    plt.colorbar(scatter, ax=ax, label=r"$\lambda_{Lm+n}(\mathbf{G}_{L,K})$")
    ax.grid(True, alpha=0.3)
    
    # Right: margin values
    ax = axes[1]
    indices = np.arange(len(margins))
    colors = ['green' if m > threshold else 'red' for m in margins]
    ax.bar(indices, margins, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.1e}')
    ax.set_xlabel(r"Candidate $\lambda$ Index")
    ax.set_ylabel(r"$\lambda_{Lm+n}(\mathbf{G}_{L,K})$")
    ax.set_title("Controllability Margins")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

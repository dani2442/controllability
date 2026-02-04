"""Visualization functions for controllability analysis.

This module provides plotting utilities for:
- Trajectory visualization (2D and 3D)
- Error convergence curves
- Matrix comparisons
- Singular value comparisons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, List, Tuple, Union


def plot_trajectory_2d(
    x: torch.Tensor,
    pairs: Optional[List[Tuple[int, int]]] = None,
    title: str = "2D Phase Portrait",
    cmap: str = "viridis",
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot 2D phase portraits of state trajectory.
    
    Args:
        x: State trajectory of shape (N, n)
        pairs: List of (i, j) pairs to plot. If None, plots consecutive pairs.
        title: Plot title
        cmap: Colormap name
        alpha: Line transparency
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    N, n = x_np.shape
    
    if pairs is None:
        pairs = [(i, i+1) for i in range(0, n-1, 2)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.9, len(pairs)))
    
    for idx, (i, j) in enumerate(pairs):
        ax.plot(x_np[:, i], x_np[:, j], color=colors[idx], alpha=alpha,
                label=f"$x_{{{i}}}, x_{{{j}}}$")
    
    ax.set_xlabel(f"State components")
    ax.set_ylabel(f"State components")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_trajectory_3d(
    x: torch.Tensor,
    triples: Optional[List[Tuple[int, int, int]]] = None,
    title: str = "3D Phase Portrait",
    cmap: str = "viridis",
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot 3D phase portraits of state trajectory.
    
    Args:
        x: State trajectory of shape (N, n)
        triples: List of (i, j, k) triples to plot. If None, plots consecutive triples.
        title: Plot title
        cmap: Colormap name
        alpha: Line transparency
        figsize: Figure size
        ax: Existing 3D axes to plot on
        
    Returns:
        fig, ax: Matplotlib figure and 3D axes
    """
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    N, n = x_np.shape
    
    if triples is None:
        triples = [(3*i, 3*i+1, 3*i+2) for i in range(n // 3)]
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.9, len(triples)))
    
    for idx, (i, j, k) in enumerate(triples):
        ax.plot(x_np[:, i], x_np[:, j], x_np[:, k], 
                color=colors[idx], alpha=alpha,
                label=f"$x_{{{i}}}, x_{{{j}}}, x_{{{k}}}$")
    
    ax.set_xlabel("$x_i$")
    ax.set_ylabel("$x_j$")
    ax.set_zlabel("$x_k$")
    ax.set_title(title)
    ax.legend(loc="best")
    # tight layout
    fig.tight_layout()
    return fig, ax


def plot_time_series(
    ts: torch.Tensor,
    x: torch.Tensor,
    components: Optional[List[int]] = None,
    title: str = "State Trajectory",
    ylabel: str = "State",
    cmap: str = "tab10",
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot state components over time.
    
    Args:
        ts: Time grid of shape (N,)
        x: State trajectory of shape (N, n)
        components: List of component indices to plot. If None, plots all.
        title: Plot title
        ylabel: Y-axis label
        cmap: Colormap name
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    ts_np = ts.cpu().numpy() if isinstance(ts, torch.Tensor) else ts
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    N, n = x_np.shape
    
    if components is None:
        components = list(range(n))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(components)))
    
    for idx, i in enumerate(components):
        ax.plot(ts_np, x_np[:, i], color=colors[idx], label=f"$x_{{{i}}}$")
    
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", ncol=min(len(components), 5))
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_error_vs_T(
    T_values: Union[List[float], np.ndarray, torch.Tensor],
    errors: Union[List[float], np.ndarray, torch.Tensor],
    theoretical_rate: float = -0.5,
    title: str = "Estimation Error vs. Horizon Length",
    xlabel: str = "$T$",
    ylabel: str = "$\\|\\hat{P}_\\lambda - P_\\lambda\\|_2$",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Axes] = None,
    label: str = "Empirical error",
) -> Tuple[Figure, Axes]:
    """Plot error convergence as function of horizon T.
    
    Shows the $T^{-1/2}$ rate from Proposition 1.
    
    Args:
        T_values: Array of horizon lengths
        errors: Corresponding error values
        theoretical_rate: Expected rate (default -0.5 for T^{-1/2})
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        ax: Existing axes to plot on
        label: Legend label for empirical data
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    T_np = np.array(T_values) if not isinstance(T_values, np.ndarray) else T_values
    err_np = np.array(errors) if not isinstance(errors, np.ndarray) else errors
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot empirical errors
    ax.loglog(T_np, err_np, 'o-', label=label, markersize=8)
    
    # Plot theoretical rate
    if theoretical_rate is not None:
        # Fit constant to match data
        C = np.mean(err_np * T_np**(-theoretical_rate))
        T_fit = np.linspace(T_np.min(), T_np.max(), 100)
        err_fit = C * T_fit**theoretical_rate
        ax.loglog(T_fit, err_fit, '--', color='red', 
                  label=f"$O(T^{{{theoretical_rate:.1f}}})$")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    return fig, ax


def plot_matrix_comparison(
    P_true: torch.Tensor,
    P_hat: torch.Tensor,
    titles: Tuple[str, str, str] = ("$P_\\lambda$ (true)", "$\\hat{P}_\\lambda$ (estimated)", "Difference"),
    figsize: Tuple[int, int] = (10, 3),
    cmap: str = "viridis",
) -> Tuple[Figure, List[Axes]]:
    """Plot comparison of true and estimated Hautus matrices.
    
    Args:
        P_true: True Hautus matrix
        P_hat: Estimated Hautus matrix
        titles: Titles for (true, estimated, difference) plots
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        fig, axs: Matplotlib figure and list of axes
    """
    P_true_np = P_true.abs().cpu().numpy() if isinstance(P_true, torch.Tensor) else np.abs(P_true)
    P_hat_np = P_hat.abs().cpu().numpy() if isinstance(P_hat, torch.Tensor) else np.abs(P_hat)
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Common colorbar limits for first two plots
    vmin = min(P_true_np.min(), P_hat_np.min())
    vmax = max(P_true_np.max(), P_hat_np.max())
    
    im0 = axs[0].imshow(P_true_np, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    axs[0].set_title(titles[0])
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    
    im1 = axs[1].imshow(P_hat_np, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    axs[1].set_title(titles[1])
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    
    # Difference plot
    diff = np.abs(P_hat_np - P_true_np)
    im2 = axs[2].imshow(diff, cmap='hot', aspect='auto')
    axs[2].set_title(titles[2])
    plt.colorbar(im2, ax=axs[2], fraction=0.046)
    
    plt.tight_layout()
    return fig, axs


def plot_singular_values(
    sigma_true: torch.Tensor,
    sigma_hat: torch.Tensor,
    title: str = "Singular Value Comparison",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot comparison of true and estimated singular values.
    
    Args:
        sigma_true: True singular values
        sigma_hat: Estimated singular values
        title: Plot title
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    sigma_true_np = sigma_true.cpu().numpy() if isinstance(sigma_true, torch.Tensor) else sigma_true
    sigma_hat_np = sigma_hat.cpu().numpy() if isinstance(sigma_hat, torch.Tensor) else sigma_hat
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    indices = np.arange(1, len(sigma_true_np) + 1)
    width = 0.35
    
    ax.bar(indices - width/2, sigma_true_np, width, label="True $\\sigma_i(P_\\lambda)$", alpha=0.8)
    ax.bar(indices + width/2, sigma_hat_np, width, label="Estimated $\\sigma_i(\\hat{P}_\\lambda)$", alpha=0.8)
    
    ax.set_xlabel("Index $i$")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(indices)
    
    return fig, ax


def plot_error_bound_comparison(
    T_values: Union[List[float], np.ndarray],
    empirical_errors: Union[List[float], np.ndarray],
    theoretical_bounds: Union[List[float], np.ndarray],
    title: str = "Empirical Error vs. Theoretical Bound",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot comparison of empirical error with theoretical bound from Proposition 1.
    
    Args:
        T_values: Array of horizon lengths
        empirical_errors: Empirical estimation errors
        theoretical_bounds: Bounds from Prop. 1
        title: Plot title
        figsize: Figure size
        ax: Existing axes to plot on
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    T_np = np.array(T_values)
    emp_np = np.array(empirical_errors)
    theo_np = np.array(theoretical_bounds)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.semilogy(T_np, emp_np, 'o-', label="Empirical error", markersize=8)
    ax.semilogy(T_np, theo_np, 's--', label="Theoretical bound (Prop. 1)", markersize=8)
    
    ax.set_xlabel("$T$")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    return fig, ax


def plot_controllability_margin(
    lambdas: torch.Tensor,
    sigma_mins: torch.Tensor,
    threshold: float = 0.0,
    title: str = "Controllability Margin at Candidate Eigenvalues",
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[Figure, Axes]:
    """Plot minimum singular values at candidate eigenvalues.
    
    Args:
        lambdas: Candidate eigenvalues (complex)
        sigma_mins: Minimum singular values at each λ
        threshold: Threshold for controllability
        title: Plot title
        figsize: Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    lambdas_np = lambdas.cpu().numpy() if isinstance(lambdas, torch.Tensor) else lambdas
    sigma_np = sigma_mins.cpu().numpy() if isinstance(sigma_mins, torch.Tensor) else sigma_mins
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot eigenvalues in complex plane
    ax1.scatter(lambdas_np.real, lambdas_np.imag, c=sigma_np, cmap='viridis', 
                s=100, edgecolors='black')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel("$\\Re(\\lambda)$")
    ax1.set_ylabel("$\\Im(\\lambda)$")
    ax1.set_title("Candidate $\\lambda$ in complex plane")
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of singular values
    indices = np.arange(1, len(sigma_np) + 1)
    colors = ['green' if s > threshold else 'red' for s in sigma_np]
    ax2.bar(indices, sigma_np, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    ax2.set_xlabel("Candidate index")
    ax2.set_ylabel("$\\sigma_{\\min}(\\hat{P}_\\lambda)$")
    ax2.set_title("Minimum singular values")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig, (ax1, ax2)

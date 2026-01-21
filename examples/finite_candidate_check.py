"""Example demonstrating finite candidate λ checking from Corollary 2.

This example shows how to:
1. Compute the candidate eigenvalues from K = S_X^{-1} M_X
2. Check controllability by testing only at candidates
3. Compare with testing at other λ values
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")

from examples import save_fig

from control import (
    ControlledLinearSDE,
    create_time_grid,
    sdeint_safe,
    compute_candidate_eigenvalues,
    hautus_test,
    compare_with_true,
    true_Hautus_matrix,
    check_controllability,
    make_stable_A,
    plot_controllability_margin,
)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(456)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System dimensions
    n, m, q = 6, 3, 2
    
    # Create system matrices (controllable pair)
    A = make_stable_A(n, device, margin=0.1)
    B = torch.randn(n, m, device=device)
    Beta = torch.randn(n, q, device=device) / n
    
    # Simulation parameters
    T = 100.0
    dt = 0.05
    batch = 1
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    x = sdeint_safe(sde, x0, ts, dt)[:, 0, :]
    u = sde.u(ts, x)
    if u.ndim == 3:
        u = u[:, 0, :]
    
    print(f"Simulated trajectory: x.shape = {x.shape}, u.shape = {u.shape}")
    
    # Compute candidate eigenvalues
    lambdas = compute_candidate_eigenvalues(x, dt)
    print(f"\n{'='*60}")
    print("CANDIDATE EIGENVALUES (from K = S_X^{-1} M_X)")
    print("="*60)
    for i, lam in enumerate(lambdas):
        print(f"  λ_{i+1} = {lam.item():.4f}")
    
    # True eigenvalues of A
    true_eigenvalues = torch.linalg.eigvals(A)
    print(f"\nTrue eigenvalues of A:")
    for i, ev in enumerate(true_eigenvalues):
        print(f"  eig_{i+1} = {ev.item():.4f}")
    
    # Test Hautus condition at each candidate
    print(f"\n{'='*60}")
    print("HAUTUS TEST AT CANDIDATE EIGENVALUES")
    print("="*60)
    
    sigma_mins_estimated = []
    sigma_mins_true = []
    
    for i, lam in enumerate(lambdas):
        result = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="time")
        sigma_mins_estimated.append(result["sigma_min_hat"])
        sigma_mins_true.append(result["sigma_min_true"])
        
        print(f"\nλ_{i+1} = {lam.item():.4f}")
        print(f"  σ_min(P_λ) true:      {result['sigma_min_true']:.6f}")
        print(f"  σ_min(P̂_λ) estimated: {result['sigma_min_hat']:.6f}")
        print(f"  Error ||P̂ - P||_2:   {result['error_norm']:.6f}")
    
    sigma_mins_estimated = torch.tensor(sigma_mins_estimated)
    sigma_mins_true = torch.tensor(sigma_mins_true)
    
    # Check controllability
    print(f"\n{'='*60}")
    print("CONTROLLABILITY CHECK (using check_controllability)")
    print("="*60)
    
    result = check_controllability(x, u, dt, ridge=1e-8, method="time", return_details=True)
    
    print(f"Is controllable: {result['is_controllable']}")
    print(f"Minimum σ_min across candidates: {result['min_sigma']:.6f}")
    
    # Compare with random λ values
    print(f"\n{'='*60}")
    print("COMPARISON WITH RANDOM λ VALUES")
    print("="*60)
    
    random_lambdas = torch.randn(5, dtype=torch.complex64, device=device) * 2
    
    for lam in random_lambdas:
        result = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="time")
        print(f"λ = {lam.item():.4f}")
        print(f"  σ_min(P_λ) true:      {result['sigma_min_true']:.6f}")
        print(f"  σ_min(P̂_λ) estimated: {result['sigma_min_hat']:.6f}")
    
    # Visualization
    fig, axs = plot_controllability_margin(
        lambdas, 
        sigma_mins_estimated,
        threshold=1e-6,
        title="Controllability Margin Analysis"
    )
    
    # Add another figure comparing true vs estimated
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    indices = np.arange(1, len(lambdas) + 1)
    width = 0.35
    
    ax1.bar(indices - width/2, sigma_mins_true.numpy(), width, label='True', alpha=0.8)
    ax1.bar(indices + width/2, sigma_mins_estimated.numpy(), width, label='Estimated', alpha=0.8)
    ax1.set_xlabel('Candidate index')
    ax1.set_ylabel('$\\sigma_{\\min}$')
    ax1.set_title('True vs. Estimated Minimum Singular Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(indices)
    
    # Plot candidate vs true eigenvalues in complex plane
    ax2.scatter(lambdas.real.cpu(), lambdas.imag.cpu(), s=100, marker='o',
                label='Candidates $\\sigma(K)$', edgecolors='black', zorder=5)
    ax2.scatter(true_eigenvalues.real.cpu(), true_eigenvalues.imag.cpu(), s=100, marker='x',
                label='True $\\sigma(A)$', linewidths=2, zorder=5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('$\\Re(\\lambda)$')
    ax2.set_ylabel('$\\Im(\\lambda)$')
    ax2.set_title('Candidate vs. True Eigenvalues')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    p = save_fig('finite_candidate_check.png', dpi=150)
    print(f"\nPlot saved to '{p}'")
    plt.show()


if __name__ == "__main__":
    main()

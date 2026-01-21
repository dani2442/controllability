"""Demo script showing usage of the control package for Hautus tests.

This script demonstrates the refactored codebase with all functionality
organized into the control/ package.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from control import (
    # SDE simulation
    ControlledLinearSDE,
    simulate_sde,
    create_time_grid,
    # Gramians
    gramian_Sz_time,
    integral_xxH_time,
    integral_xdot_xH_time,
    compute_candidate_eigenvalues,
    # Hautus test
    cross_moment_H_time,
    cross_moment_H_fft,
    estimate_Hautus_matrix,
    true_Hautus_matrix,
    compare_with_true,
    check_controllability,
    # Utilities
    make_stable_A,
    # Visualization
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_matrix_comparison,
)


def main():
    """Main demo function."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"Device: {device}")
    
    # System parameters
    T = 100.0
    dt = 0.1
    n, m, q = 10, 5, 2
    batch = 1
    
    # Create stable system
    A = make_stable_A(n, device, margin=0.1)
    B = torch.randn(n, m, device=device)
    Beta = torch.randn((n, q), device=device) / n
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    x, u = simulate_sde(sde, x0, ts, dt)
    x = x[:, 0, :]  # (N, n)
    u = u[:, 0, :]  # (N, m)
    
    print(f"x.shape = {x.shape}")
    print(f"u.shape = {u.shape}")
    
    # Plot 2D trajectory
    fig1, ax1 = plot_trajectory_2d(x, cmap='hsv', title="2D Phase Portrait")
    
    # Plot 3D trajectory
    fig2, ax2 = plot_trajectory_3d(x, cmap='hsv', title="3D Phase Portrait")
    
    # Test Hautus condition at a specific λ
    lam = 0.3 + 1.3j
    
    # Time-domain method
    H1 = cross_moment_H_time(x, u, lam=lam, dt=dt)
    Sz = gramian_Sz_time(x, u, dt=dt)
    P1 = estimate_Hautus_matrix(H1, Sz, ridge=1e-8)
    
    # FFT method
    H2, Sz2 = cross_moment_H_fft(x, u, lam=lam, dt=dt, omega_max=100)
    P2 = estimate_Hautus_matrix(H2, Sz2, ridge=1e-8)
    
    # True Hautus matrix
    P = true_Hautus_matrix(A, B, lam)
    
    # Compare singular values
    P_svd = torch.linalg.svdvals(P)
    P1_svd = torch.linalg.svdvals(P1)
    P2_svd = torch.linalg.svdvals(P2)
    
    print(f"\n{'='*50}")
    print(f"HAUTUS TEST AT λ = {lam}")
    print("="*50)
    print(f"σ_min(P) (true)          = {P_svd.min().item():.6f}")
    print(f"σ_min(P̂) (time-domain)   = {P1_svd.min().item():.6f}")
    print(f"σ_min(P̂) (FFT)           = {P2_svd.min().item():.6f}")
    
    # Matrix comparison visualization
    fig3, axs3 = plot_matrix_comparison(
        P, P1,
        titles=("$P_\\lambda$ (true)", "$\\hat{P}_\\lambda$ (time)", "|Difference|")
    )
    plt.suptitle(f"$\\lambda = {lam}$")
    
    # Compute candidate eigenvalues (Theorem 4)
    Sx = integral_xxH_time(x, dt)
    Mx = integral_xdot_xH_time(x, dt)
    K = torch.linalg.solve(Sx, Mx.T).T
    lambdas = torch.linalg.eigvals(K)
    
    print(f"\n{'='*50}")
    print("CANDIDATE EIGENVALUES")
    print("="*50)
    for i, l in enumerate(lambdas):
        print(f"  λ_{i+1} = {l.item():.4f}")
    
    # Check controllability at all candidate eigenvalues
    print(f"\n{'='*50}")
    print("CONTROLLABILITY CHECK AT CANDIDATES")
    print("="*50)
    
    min_eval = np.inf
    min_eval1 = np.inf
    min_eval2 = np.inf
    
    for lam in lambdas:
        # Compare with true
        result = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="time")
        result_fft = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="fft", omega_max=100)
        
        min_eval = min(min_eval, result["sigma_min_true"])
        min_eval1 = min(min_eval1, result["sigma_min_hat"])
        min_eval2 = min(min_eval2, result_fft["sigma_min_hat"])
    
    print(f"min σ_min (true):        {min_eval:.6f}")
    print(f"min σ_min (time-domain): {min_eval1:.6f}")
    print(f"min σ_min (FFT):         {min_eval2:.6f}")
    
    if min_eval > 1e-6:
        print("\n✓ System is controllable (all singular values > 0)")
    else:
        print("\n✗ System may not be controllable")
    
    plt.show()


if __name__ == "__main__":
    main()

"""Basic example demonstrating the data-driven Hautus test.

This example shows how to:
1. Simulate a controlled linear SDE
2. Compute the estimated Hautus matrix from data
3. Compare with the true Hautus matrix
4. Visualize the results
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import (
    ControlledLinearSDE,
    create_time_grid,
    simulate_sde,
    compare_with_true,
    hautus_test,
    true_Hautus_matrix,
    make_stable_A,
    plot_trajectory_2d,
    plot_matrix_comparison,
    plot_singular_values,
)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System dimensions
    n, m, q = 6, 3, 2  # state, input, noise dimensions
    
    # Create system matrices
    A = make_stable_A(n, device, margin=0.1)
    B = torch.randn(n, m, device=device)
    Beta = torch.randn(n, q, device=device) / n  # noise coefficient
    
    # Simulation parameters
    T = 50.0
    dt = 0.05
    batch = 1
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    x, u = simulate_sde(sde, x0, ts, dt)
    x = x[:, 0, :]  # Remove batch dimension (N, n)
    u = u[:, 0, :]  # (N, m)
    
    print(f"Simulated trajectory: x.shape = {x.shape}, u.shape = {u.shape}")
    
    # Choose a test value of lambda
    lam = 0.3 + 1.3j
    
    # Compute the Hautus test with time-domain method
    result_time = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="time")
    
    # Compute the Hautus test with FFT method
    result_fft = compare_with_true(A, B, x, u, dt, lam, ridge=1e-8, method="fft", omega_max=100)
    
    print("\n" + "="*60)
    print("HAUTUS TEST RESULTS")
    print("="*60)
    print(f"λ = {lam}")
    print(f"\nTrue σ_min(P_λ) = {result_time['sigma_min_true']:.6f}")
    print(f"\nTime-domain method:")
    print(f"  Estimated σ_min(P̂_λ) = {result_time['sigma_min_hat']:.6f}")
    print(f"  Error ||P̂_λ - P_λ||_2 = {result_time['error_norm']:.6f}")
    print(f"\nFFT method:")
    print(f"  Estimated σ_min(P̂_λ) = {result_fft['sigma_min_hat']:.6f}")
    print(f"  Error ||P̂_λ - P_λ||_2 = {result_fft['error_norm']:.6f}")
    
    # Visualize trajectory
    fig1, ax1 = plot_trajectory_2d(
        x, 
        title=f"State Trajectory (T={T}, n={n})",
        cmap="viridis"
    )
    
    # Compare matrices
    fig2, axs2 = plot_matrix_comparison(
        result_time["P_true"],
        result_time["P_hat"],
        titles=("$P_\\lambda$ (true)", "$\\hat{P}_\\lambda$ (time)", "|Difference|"),
    )
    plt.suptitle(f"Time-domain method, λ = {lam}")
    
    # Compare singular values
    sigma_true = torch.linalg.svdvals(result_time["P_true"])
    sigma_hat_time = torch.linalg.svdvals(result_time["P_hat"])
    sigma_hat_fft = torch.linalg.svdvals(result_fft["P_hat"])
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    indices = range(1, len(sigma_true) + 1)
    width = 0.25
    
    ax3.bar([i - width for i in indices], sigma_true.cpu().numpy(), width, 
            label="True", alpha=0.8)
    ax3.bar([i for i in indices], sigma_hat_time.abs().cpu().numpy(), width,
            label="Time-domain", alpha=0.8)
    ax3.bar([i + width for i in indices], sigma_hat_fft.abs().cpu().numpy(), width,
            label="FFT", alpha=0.8)
    
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Singular value")
    ax3.set_title(f"Singular Values Comparison, λ = {lam}")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

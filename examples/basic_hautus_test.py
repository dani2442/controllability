"""Basic example demonstrating the data-driven Hautus test.

This example shows how to:
1. Simulate a controlled linear SDE
2. Compute the estimated Hautus matrix from data
3. Compare with the true Hautus matrix
4. Visualize the results including error distribution across eigenvalues
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import save_fig

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
    compute_candidate_eigenvalues,
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
    T = 100.0
    dt = 0.05
    batch = 50  # Use larger batch for violinplot samples
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    x_full, u_full = simulate_sde(sde, x0, ts, dt)  # (N, batch, n) and (N, batch, m)
    
    # For single-trajectory analysis, use first batch element
    x = x_full[:, 0, :]  # (N, n)
    u = u_full[:, 0, :]  # (N, m)
    
    print(f"Simulated {batch} trajectories: x_full.shape = {x_full.shape}")
    
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
    
    # =========================================================================
    # Test multiple eigenvalues for violinplot (using all batch samples)
    # =========================================================================
    print("\n" + "="*60)
    print("TESTING MULTIPLE EIGENVALUES")
    print("="*60)
    
    # Get candidate eigenvalues from the system matrix A
    eigenvalues_A = torch.linalg.eigvals(A).cpu().numpy()
    
    # Also test some random eigenvalues in the left half-plane
    np.random.seed(42)
    n_random = 20
    random_lambdas = -np.abs(np.random.randn(n_random)) + 1j * np.random.randn(n_random) * 2
    
    # Combine: actual eigenvalues + random test points
    test_lambdas = list(eigenvalues_A) + list(random_lambdas)
    
    # Collect errors for each eigenvalue and each batch sample
    records = []
    for b in range(batch):
        x_b = x_full[:, b, :]  # (N, n)
        u_b = u_full[:, b, :]  # (N, m)
        
        for i, lam_test in enumerate(test_lambdas):
            lam_test = complex(lam_test)
            
            result_time = compare_with_true(A, B, x_b, u_b, dt, lam_test, ridge=1e-8, method="time")
            result_fft = compare_with_true(A, B, x_b, u_b, dt, lam_test, ridge=1e-8, method="fft", omega_max=100)
            
            is_eig = "A eigenvalue" if i < len(eigenvalues_A) else "Random λ"
            
            # Use σ_min(P) - σ_min(P̂) as error (signed difference)
            records.append({
                "λ": f"{lam_test:.2f}",
                "batch": b,
                "type": is_eig,
                "method": "Time-domain",
                "error": result_time["sigma_min_true"] - result_time["sigma_min_hat"],
                "sigma_min_hat": result_time["sigma_min_hat"],
                "sigma_min_true": result_time["sigma_min_true"],
                "real_part": lam_test.real,
            })
            records.append({
                "λ": f"{lam_test:.2f}",
                "batch": b,
                "type": is_eig,
                "method": "FFT",
                "error": result_fft["sigma_min_true"] - result_fft["sigma_min_hat"],
                "sigma_min_hat": result_fft["sigma_min_hat"],
                "sigma_min_true": result_fft["sigma_min_true"],
                "real_part": lam_test.real,
            })
    
    df = pd.DataFrame(records)
    
    print(f"Tested {len(test_lambdas)} eigenvalues × {batch} samples = {len(test_lambdas) * batch} total")
    print(f"  ({len(eigenvalues_A)} from A, {n_random} random)")
    print(f"Mean error (Time): {df[df['method']=='Time-domain']['error'].mean():.6f}")
    print(f"Mean error (FFT):  {df[df['method']=='FFT']['error'].mean():.6f}")
    
    # =========================================================================
    # Violinplot for errors
    # =========================================================================
    sns.set_theme(style="whitegrid")
    
    fig_violin, ax1 = plt.subplots(figsize=(8, 5))
    
    # Violinplot: Errors by method
    sns.violinplot(
        data=df, x="method", y="error", hue="type",
        split=True, inner="quart", palette="muted", ax=ax1
    )
    ax1.set_xlabel("Method")
    ax1.set_ylabel("$\\sigma_{\\min}(P_\\lambda) - \\sigma_{\\min}(\\hat{P}_\\lambda)$")
    ax1.set_title("Error Distribution by Method and λ Type")
    ax1.legend(title="λ type", loc="upper right")
    
    plt.tight_layout()
    p1 = save_fig('hautus_error_violinplot.png', fig=fig_violin, dpi=150)
    print(f"\nViolinplot saved to '{p1}'")
    
    # =========================================================================
    # Original visualizations
    # =========================================================================
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

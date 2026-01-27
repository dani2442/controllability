"""Example demonstrating the T^{-1/2} error convergence rate.

This example validates Proposition 1 (Cross-moment error bound) by:
1. Running simulations for different horizon lengths T
2. Computing the estimation error ||P̂_λ - P_λ||_2
3. Verifying the O(T^{-1/2}) convergence rate
4. Comparing with the theoretical bound
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import save_fig

from control import (
    ControlledLinearSDE,
    create_time_grid,
    sdeint_safe,
    compare_with_true,
    make_stable_A,
    plot_error_vs_T,
    plot_error_bound_comparison,
)


def compute_theoretical_bound(
    T: float,
    beta_norm: float,
    sigma_min_Sz_normalized: float,
    q: int,
    n: int,
    m: int,
    delta: float = 0.05,
) -> float:
    """Compute the theoretical error bound from Proposition 1.
    
    ||P̂_λ - P_λ||_2 ≤ (||β|| / √(T σ_min(S̄_Z))) * (√q + √(n+m) + √(2 log(1/δ)))
    
    Args:
        T: Horizon length
        beta_norm: ||β||_2
        sigma_min_Sz_normalized: σ_min(S̄_Z(T)) = σ_min(S_Z(u)/T)
        q: Noise dimension
        n: State dimension
        m: Input dimension
        delta: Confidence level (default 0.05 for 95% confidence)
        
    Returns:
        Theoretical upper bound on the error
    """
    constant_term = np.sqrt(q) + np.sqrt(n + m) + np.sqrt(2 * np.log(1 / delta))
    return beta_norm / np.sqrt(T * sigma_min_Sz_normalized) * constant_term


def main():
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System dimensions
    n, m, q = 5, 3, 2
    
    # Create system matrices
    A = make_stable_A(n, device, margin=0.2)
    B = torch.randn(n, m, device=device)
    Beta = torch.randn(n, q, device=device) * 0.5  # Moderate noise
    
    beta_norm = torch.linalg.norm(Beta, ord=2).item()
    
    # Simulation parameters
    dt = 0.05
    batch = 50  # Use batch for Monte Carlo trials
    
    # Test value of lambda
    lam = 0.5 + 0.8j
    
    # Range of horizon lengths - simulate once with max T and subsample
    T_values = [10, 20, 50, 100, 200, 500, 1000]
    T_max = max(T_values)
    
    print(f"System: n={n}, m={m}, q={q}")
    print(f"Testing λ = {lam}")
    print(f"Simulating batch of {batch} trajectories with T_max = {T_max}...")
    
    # Create SDE and simulate once with maximum T
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T_max, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    # Simulate all trajectories at once: shape (N, batch, n)
    x_full = sdeint_safe(sde, x0, ts, dt)
    u_full = sde.u(ts, x_full[:, 0, :])  # (N, m)
    if u_full.ndim == 3:
        u_full = u_full[:, 0, :]  # Control is same for all batch elements
    
    # Storage for results - collect all individual errors for seaborn
    from control import gramian_Sz_time
    
    records = []  # For DataFrame
    mean_errors_time = []
    mean_errors_fft = []
    mean_errors_laplace = []
    std_errors_time = []
    std_errors_fft = []
    std_errors_laplace = []
    theoretical_bounds = []
    
    for T in T_values:
        print(f"\nT = {T}...")
        
        # Compute number of timesteps for this T
        N_T = int(T / dt) + 1
        
        # Subsample trajectories up to time T
        x_T = x_full[:N_T]  # (N_T, batch, n)
        u_T = u_full[:N_T]  # (N_T, m)
        
        errors_time = []
        errors_fft = []
        errors_laplace = []
        sigma_min_Sz_list = []
        
        # Process each batch element
        for b in range(batch):
            x_b = x_T[:, b, :]  # (N_T, n)
            
            # Compute errors
            result_time = compare_with_true(A, B, x_b, u_T, dt, lam, ridge=1e-10, method="time")
            result_fft = compare_with_true(A, B, x_b, u_T, dt, lam, ridge=1e-10, method="fft", omega_max=100)
            result_laplace = compare_with_true(A, B, x_b, u_T, dt, lam, ridge=1e-10, method="laplace", sigma=0.1, omega_max=100)
            
            err_time = result_time["error_norm"]
            err_fft = result_fft["error_norm"]
            err_laplace = result_laplace["error_norm"]
            
            errors_time.append(err_time)
            errors_fft.append(err_fft)
            errors_laplace.append(err_laplace)
            
            # Record for DataFrame
            records.append({"T": T, "batch": b, "error_time": err_time, "error_fft": err_fft, "error_laplace": err_laplace})
            
            # Compute normalized S_Z for theoretical bound
            Sz = gramian_Sz_time(x_b, u_T, dt)
            sigma_min_Sz_list.append(torch.linalg.svdvals(Sz).min().item() / T)
        
        mean_errors_time.append(np.mean(errors_time))
        mean_errors_fft.append(np.mean(errors_fft))
        mean_errors_laplace.append(np.mean(errors_laplace))
        std_errors_time.append(np.std(errors_time))
        std_errors_fft.append(np.std(errors_fft))
        std_errors_laplace.append(np.std(errors_laplace))
        
        # Compute theoretical bound (using average sigma_min)
        avg_sigma_min = np.mean(sigma_min_Sz_list)
        bound = compute_theoretical_bound(T, beta_norm, avg_sigma_min, q, n, m)
        theoretical_bounds.append(bound)
        
        print(f"  Time:    {mean_errors_time[-1]:.4f} ± {std_errors_time[-1]:.4f}")
        print(f"  FFT:     {mean_errors_fft[-1]:.4f} ± {std_errors_fft[-1]:.4f}")
        print(f"  Laplace: {mean_errors_laplace[-1]:.4f} ± {std_errors_laplace[-1]:.4f}")
        print(f"  Bound:   {bound:.4f}")
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(records)
    
    # Convert to arrays for classic plots
    T_values_arr = np.array(T_values)
    mean_errors_time = np.array(mean_errors_time)
    mean_errors_fft = np.array(mean_errors_fft)
    theoretical_bounds = np.array(theoretical_bounds)
    
    # =========================================================================
    # Classic plot: Theoretical bound comparison with individual trajectories
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    # Plot individual batch trajectories with transparency (connected over T)
    for b in range(batch):
        errors_b_time = df[df['batch'] == b]['error_time'].values
        errors_b_fft = df[df['batch'] == b]['error_fft'].values
        ax.plot(T_values_arr, errors_b_time, '-', color='C0', alpha=0.15, linewidth=0.8)
        ax.plot(T_values_arr, errors_b_fft, '-', color='C1', alpha=0.15, linewidth=0.8)
    
    # Mean lines with markers
    ax.loglog(T_values_arr, mean_errors_time, 'o-', label='Empirical (time)', markersize=8, color='C0')
    ax.loglog(T_values_arr, mean_errors_fft, 's-', label='Empirical (FFT)', markersize=8, color='C1')
    ax.loglog(T_values_arr, theoretical_bounds, '^--', label='Theoretical bound', markersize=8, color='C2')
    
    # Fill between for std (time method)
    ax.fill_between(T_values_arr, 
                      mean_errors_time - np.array(std_errors_time),
                      mean_errors_time + np.array(std_errors_time),
                      alpha=0.3, color='C0')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Horizon length $T$')
    ax.set_ylabel('Error')
    ax.set_title('Empirical Error vs. Theoretical Bound')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    p2 = save_fig('paper/images/error_convergence.pdf', dpi=150)
    print(f"Classic plot saved to '{p2}'")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Estimate convergence rate
    log_T = np.log(T_values_arr)
    log_err = np.log(mean_errors_time)
    slope, intercept = np.polyfit(log_T, log_err, 1)
    
    print(f"Estimated convergence rate: T^{{{slope:.3f}}}")
    print(f"Expected rate: T^{{-0.5}}")
    print(f"Rate error: {abs(slope + 0.5):.3f}")


if __name__ == "__main__":
    main()

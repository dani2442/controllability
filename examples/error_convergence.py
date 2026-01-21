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
import sys
sys.path.insert(0, "..")

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
    batch = 1
    
    # Test value of lambda
    lam = 0.5 + 0.8j
    
    # Range of horizon lengths
    T_values = [10, 20, 50, 100, 200, 500, 1000]
    
    # Number of Monte Carlo trials
    n_trials = 10
    
    print(f"System: n={n}, m={m}, q={q}")
    print(f"Testing λ = {lam}")
    print(f"Running {n_trials} trials for each T value...")
    
    # Storage for results
    mean_errors_time = []
    mean_errors_fft = []
    std_errors_time = []
    std_errors_fft = []
    theoretical_bounds = []
    
    for T in T_values:
        print(f"\nT = {T}...")
        
        errors_time = []
        errors_fft = []
        sigma_min_Sz_list = []
        
        for trial in range(n_trials):
            # Create SDE and simulate
            sde = ControlledLinearSDE(A, B, Beta).to(device)
            ts = create_time_grid(T, dt, device)
            x0 = torch.zeros(batch, n, device=device)
            
            # Simulate
            x = sdeint_safe(sde, x0, ts, dt)[:, 0, :]
            u = sde.u(ts, x)
            if u.ndim == 3:
                u = u[:, 0, :]
            
            # Compute errors
            result_time = compare_with_true(A, B, x, u, dt, lam, ridge=1e-10, method="time")
            result_fft = compare_with_true(A, B, x, u, dt, lam, ridge=1e-10, method="fft", omega_max=100)
            
            errors_time.append(result_time["error_norm"])
            errors_fft.append(result_fft["error_norm"])
            
            # Compute normalized S_Z for theoretical bound
            from control import gramian_Sz_time
            Sz = gramian_Sz_time(x, u, dt)
            sigma_min_Sz_list.append(torch.linalg.svdvals(Sz).min().item() / T)
        
        mean_errors_time.append(np.mean(errors_time))
        mean_errors_fft.append(np.mean(errors_fft))
        std_errors_time.append(np.std(errors_time))
        std_errors_fft.append(np.std(errors_fft))
        
        # Compute theoretical bound (using average sigma_min)
        avg_sigma_min = np.mean(sigma_min_Sz_list)
        bound = compute_theoretical_bound(T, beta_norm, avg_sigma_min, q, n, m)
        theoretical_bounds.append(bound)
        
        print(f"  Time: {mean_errors_time[-1]:.4f} ± {std_errors_time[-1]:.4f}")
        print(f"  FFT:  {mean_errors_fft[-1]:.4f} ± {std_errors_fft[-1]:.4f}")
        print(f"  Bound: {bound:.4f}")
    
    # Convert to arrays
    T_values = np.array(T_values)
    mean_errors_time = np.array(mean_errors_time)
    mean_errors_fft = np.array(mean_errors_fft)
    theoretical_bounds = np.array(theoretical_bounds)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs T (log-log)
    ax1.loglog(T_values, mean_errors_time, 'o-', label='Time-domain', markersize=8)
    ax1.loglog(T_values, mean_errors_fft, 's-', label='FFT', markersize=8)
    
    # Fit T^{-1/2} line
    C_fit = np.mean(mean_errors_time * np.sqrt(T_values))
    T_fit = np.linspace(T_values.min(), T_values.max(), 100)
    ax1.loglog(T_fit, C_fit / np.sqrt(T_fit), '--', color='red', 
               label='$O(T^{-1/2})$', linewidth=2)
    
    ax1.set_xlabel('Horizon length $T$')
    ax1.set_ylabel('$\\|\\hat{P}_\\lambda - P_\\lambda\\|_2$')
    ax1.set_title('Error Convergence Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Comparison with theoretical bound
    ax2.semilogy(T_values, mean_errors_time, 'o-', label='Empirical (time)', markersize=8)
    ax2.semilogy(T_values, theoretical_bounds, 's--', label='Theoretical bound (Prop. 1)', markersize=8)
    ax2.fill_between(T_values, 
                      mean_errors_time - np.array(std_errors_time),
                      mean_errors_time + np.array(std_errors_time),
                      alpha=0.3)
    
    ax2.set_xlabel('Horizon length $T$')
    ax2.set_ylabel('Error')
    ax2.set_title('Empirical Error vs. Theoretical Bound')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    p = save_fig('error_convergence.png', dpi=150)
    print(f"\nPlot saved to '{p}'")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Estimate convergence rate
    log_T = np.log(T_values)
    log_err = np.log(mean_errors_time)
    slope, intercept = np.polyfit(log_T, log_err, 1)
    
    print(f"Estimated convergence rate: T^{{{slope:.3f}}}")
    print(f"Expected rate: T^{{-0.5}}")
    print(f"Rate error: {abs(slope + 0.5):.3f}")


if __name__ == "__main__":
    main()
